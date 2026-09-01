using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NastDeskview;

public class Intrinsics
{
    public double fx { get; set; }
    public double fy { get; set; }
    public double cx { get; set; }
    public double cy { get; set; }
    public int w { get; set; }
    public int h { get; set; }
}

public class MetaDto
{
    public Intrinsics intrinsics { get; set; }
    public double extent { get; set; }
    public long count { get; set; }
    public int nframes { get; set; }
}

public class FramePose
{
    public double[] q { get; set; }
    public double[] p { get; set; }
}

public class Pose
{
    public double[] t { get; set; }
    public double[] q { get; set; }
    public double[] size { get; set; }
    [JsonExtensionData] public Dictionary<string, JsonElement> Extra { get; set; }
}

public class ObjDto
{
    public int id { get; set; }
    public string label { get; set; }
    public string frame { get; set; }
    public string cam { get; set; }
    public string kind { get; set; }
    public string pts { get; set; }
    public int npts { get; set; }
    public string pose { get; set; }
    public string obs { get; set; }

    [JsonIgnore]
    public Pose P => string.IsNullOrEmpty(pose) || pose == "null"
        ? null : JsonSerializer.Deserialize<Pose>(pose);
    [JsonIgnore]
    public int ObsCount
    {
        get
        {
            try { return string.IsNullOrEmpty(obs) ? 1 : JsonDocument.Parse(obs).RootElement.GetArrayLength(); }
            catch { return 1; }
        }
    }

    public class View
    {
        public string frame; public string kind; public double[][] pts; public bool auto;
    }

    // every observation of this object (operator's + auto-collected)
    [JsonIgnore]
    public List<View> Views
    {
        get
        {
            var list = new List<View>();
            try
            {
                if (string.IsNullOrEmpty(obs)) return list;
                foreach (var e in JsonDocument.Parse(obs).RootElement.EnumerateArray())
                {
                    var v = new View
                    {
                        frame = e.GetProperty("frame").GetString(),
                        kind = e.GetProperty("kind").GetString(),
                        pts = e.GetProperty("pts").EnumerateArray()
                              .Select(q => q.EnumerateArray().Select(x => x.GetDouble()).ToArray()).ToArray(),
                        auto = e.TryGetProperty("auto", out var a) && a.ValueKind == JsonValueKind.True,
                    };
                    list.Add(v);
                }
            }
            catch { }
            return list;
        }
    }
    [JsonIgnore] public int AutoCount => Views.Count(v => v.auto);
}

public class JobDto
{
    public int id { get; set; }
    public int object_id { get; set; }
    public string kind { get; set; }
    public string status { get; set; }
    public string detail { get; set; }
}

// Thin client for the Inspector service (the 3D brain stays in python).
public class Api
{
    public string Base = "http://127.0.0.1:8130";
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(120) };
    public bool Alive { get; private set; }

    public async Task<MetaDto> Meta()
    {
        try
        {
            var m = await _http.GetFromJsonAsync<MetaDto>(Base + "/api/meta");
            Alive = m != null;
            return m;
        }
        catch { Alive = false; return null; }
    }

    public async Task<Dictionary<string, FramePose>> Poses()
        => await _http.GetFromJsonAsync<Dictionary<string, FramePose>>(Base + "/api/poses");

    public async Task<List<ObjDto>> Objects()
        => await _http.GetFromJsonAsync<List<ObjDto>>(Base + "/api/objects");

    public async Task<List<JobDto>> Jobs()
        => await _http.GetFromJsonAsync<List<JobDto>>(Base + "/api/jobs");

    public async Task<JsonElement> GetJson(string path)
    {
        var s = await _http.GetStringAsync(Base + path);
        return JsonDocument.Parse(s).RootElement;
    }

    public async Task<(bool ok, JsonElement body)> Post(string path, object payload)
    {
        // явний StringContent: stdlib-сервер читає тіло за Content-Length,
        // а PostAsJsonAsync шле chunked і залишає його порожнім
        var json = JsonSerializer.Serialize(payload);
        var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
        var r = await _http.PostAsync(Base + path, content);
        var doc = JsonDocument.Parse(await r.Content.ReadAsStringAsync());
        return (r.IsSuccessStatusCode, doc.RootElement);
    }

    public async Task Delete(string path) => await _http.DeleteAsync(Base + path);
}
