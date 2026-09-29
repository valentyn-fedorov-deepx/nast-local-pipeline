# Auth in the Inspector — developer notes

*Read this before adding or changing a route, or a client call to the service.
How login works today, how to put a route behind it, and the trap that breaks
the clients if you do it the obvious way. The operator side (creating accounts)
is in the README, "Accounts and login".*

---

## 1. What exists today

One route needs a login. Everything else is open.

| Route | Needs a login |
|---|---|
| `POST /api/open_dataset` | **yes** |
| `POST /api/auth/login` | no (it is how you get one) |
| everything else, every GET, DELETE and the other POSTs | no |

The login contract:

```
POST /api/auth/login     {"username": "...", "password": "..."}
  200  {"token": "<token>"}
  401  {"error": "invalid credentials"}   wrong password, or no such user
  400  {"error": "bad request"}           body not JSON, not an object, or fields not non-empty strings
  503  {"error": "no accounts yet: run python inspector/auth.py add-user <name>"}

a gated route
  Authorization: Bearer <token>
  401  {"error": "unauthorized"}          missing, malformed, tampered or expired token
```

**Clients key on the error strings, not the status codes.** `invalid credentials`
means *the human mistyped*: show it and let them retry. `unauthorized` means *the
session is gone*: get a new token and retry once. Both are 401, so a client that
only looks at the code cannot tell them apart. Keep these exact strings; they are
also what the board's control API uses, so every client behaves the same way.

## 2. How it works — `inspector/auth.py`

Standard library only, like the rest of the service.

- **Accounts** are in `inspector/users.json` (gitignored), one PBKDF2-SHA256 hash
  per name, 600,000 iterations, salted. Never the password. Created with
  `python inspector/auth.py add-user <name>`. `check_login()` re-reads the file on
  every attempt, so a new account works without restarting the service.
- **A token** is `base64url(claims).base64url(HMAC-SHA256)`, claims
  `{"sub", "iat", "exp"}`, valid for `TOKEN_TTL` (8 hours). It is **signed, not
  encrypted**: anyone holding it can read the claims, nobody can change them. Put
  nothing secret in the claims.
- **The signing key** is made fresh at every service start and never written to
  disk. Consequence: **a restart signs everyone out.** Clients handle that with the
  same retry as an expired token.
- `check_token(header)` returns the account name, or `None` for anything wrong
  with the header. It never raises, whatever the header contains.

## 3. Putting a route behind the login

The whole pattern, exactly as `/api/open_dataset` uses it in `server.py`:

```python
        if p == "/api/your_route":
            if auth.check_token(self.headers.get("Authorization")) is None:
                return self._send(401, {"error": "unauthorized"})
            ...                                  # the route's real work
```

Two rules:

1. **The check comes first, before any work or side effect.** A route that
   starts a job and *then* refuses has already done the damage.
2. **The check belongs in the service, never only in a client.** A login dialog
   in an app is UX; the service's refusal is the security. Anything the app
   refuses to do, `curl` can still ask the service for.

It works the same in `do_GET`, `do_POST` and `do_DELETE`. **But read §4 before
gating anything a client already calls**, because right now most client calls do
not send a token.

## 4. The trap: most client calls send no token

Only one client helper attaches the token today. Gate a route that one of the
others calls, and that client breaks with a 401 it does not know how to handle.

| Client | Sends the token | Does not |
|---|---|---|
| Qt app, `linux_app/deskview_qt.py` | `api_post` | `api_get`, `api_delete`, and the direct `urlopen` of `/frames/live/…` |
| Windows app, `deskview/` | nothing yet | every call (`Api.Post`, the GET helpers, `Delete`) |
| Browser page, `inspector/static/app.html` | nothing | every `fetch`: it has no login at all |

So before gating a route, find every caller:

```
grep -rn "your_route" linux_app deskview inspector/static
```

then, per client:

- **Qt app.** Make the helper it goes through send the header, the same way
  `api_post` does (`if TOKEN: headers["Authorization"] = f"Bearer {TOKEN}"`). If
  the call happens because the operator clicked something, ask for a login first
  with `self.ensure_login()`. If a session can lapse under it, copy the
  retry-once shape of `_post_open`: on a 401, clear `TOKEN`, log in again, retry
  once, and give up if the operator cancels.
- **Windows app.** It has no login yet. It needs an `Api.Token` set by a login
  window, attached to every request that reaches a gated route.
- **Browser page.** It cannot use the Qt or Windows login. It needs its own login
  page, and a cookie set by the login route (`HttpOnly`, `SameSite=Strict`) so its
  existing `fetch` calls carry the session unchanged. `check_token` would then also
  have to accept the cookie. This is real work: do it once, for the whole page,
  rather than route by route.

## 5. If the goal becomes "everything needs a login"

That is the likely next step, and it is not a route-by-route job. Move the check
to the top of `do_GET`, `do_POST` and `do_DELETE`, with an allowlist of the few
things that must stay open: `/api/auth/login`, and the login page with its assets.
Default-deny that way means a route added later is covered without anyone
remembering to add the check.

It only works after §4 is done for **all three** clients. Until then it locks out
the browser page completely, and the apps' GET calls fail at startup.

## 6. Testing a change

- Logic in `auth.py`: add a case to `inspector/test_auth.py` and run
  `cd inspector && python -m unittest test_auth -v`. It takes a few seconds: every
  password hash is deliberately slow.
- A newly gated route: run the service on a spare port
  (`python inspector/server.py 8131`) and check all three answers.

```bash
U=http://127.0.0.1:8131
curl -s -w ' [%{http_code}]\n' -X POST -d '{}' $U/api/your_route              # want 401 unauthorized
TOKEN=$(curl -s -X POST -d '{"username":"<name>","password":"<pw>"}' $U/api/auth/login | python -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -s -w ' [%{http_code}]\n' -X POST -H "Authorization: Bearer $TOKEN" -d '{}' $U/api/your_route   # want the route's normal answer
curl -s -o /dev/null -w '[%{http_code}]\n' $U/api/meta                         # want 200: nothing else broke
```

  Then open every client that calls the route (§4) and use it.

## 7. Do not

- **Log a token or a password**, anywhere: a token in a log is a working session
  until it expires.
- **Store a password, or anything derived from it other than `hash_password()`'s
  output.**
- **Put anything secret in the token claims.** They are readable.
- **Invent new error strings** for the same two situations. Clients match on
  `unauthorized` and `invalid credentials`.
- **Treat the loopback bind as protection.** The service listens on
  `127.0.0.1` over plain HTTP. That keeps it off the network, but any process or
  user on the same machine, and any web page open in the operator's browser, can
  reach it. If Mode 3 is ever reached from another machine, it needs TLS in front
  before anything else.
