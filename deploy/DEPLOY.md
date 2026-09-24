# NAST Mode 3 на новий ноут (Linux, RTX 5090): перенос і розгортання

## Що переносимо (одна папка, 19 ГБ)

| файл | навіщо | розмір |
|---|---|---|
| `nast_preflight.sh` | інвентар ноута і нового запису, вердикт по місцю | 6 КБ |
| `install_mode3_clean.sh` | інсталер: пакети, код, venv, TRELLIS-env, ваги, перевірка сервісу | 6 КБ |
| `nast_v3_clean_code.tar` | код (git archive, без даних) | 1.9 МБ |
| `vggt_omega_1b_512.pt` | ваги VGGT-Omega: пози, глибина, карта | 4.4 ГБ |
| `nast_trellis_pack_blackwell.tar` + `.md5` | готовий TRELLIS-env для sm_120 (той, що прогнано у WSL): SAM, Real-ESRGAN, TRELLIS, MoGe-2 | 6.3 ГБ |
| `nast_weights.tar` | ваги TRELLIS, SAM, MoGe-2, DINOv2, щоб нічого не качати на ноуті | 8.0 ГБ |
| `manifest.md5` | контрольні суми всього списку | |
| `DEPLOY.md` | ця інструкція | |
| `bootstrap.sh` | на ноуті: скачати з Drive, звірити, поставити, одною командою | |

Без `nast_weights.tar` інсталер сам скачає ваги (8 ГБ з HuggingFace і GitHub). Без TRELLIS-паку об'єкти будуть вирізатись із точок карти, без генеративного меша.

## Що має бути на ноуті до початку

- Ubuntu 22.04 або 24.04, звичайний користувач із sudo (не root).
- Драйвер NVIDIA: `nvidia-smi` працює, для RTX 50xx потрібна версія 570 або новіша з open-модулями (`nvidia-driver-570-open`). Драйвер інсталер НЕ ставить.
- Місце: 110 ГБ вільних на диску, де буде `~/nast` (45 ГБ ставиться плюс стільки ж на час розпакування), і 10 МБ на кожен кадр запису.
- Інтернет: потрібен для apt і pip (torch cu128 близько 3 ГБ). Ваги з бандла, без інтернету.

## Найкоротший шлях: одна команда з Google Drive

Кіт лежить на Drive у `nast_demo/mode3_5090`. На ноуті потрібен rclone з цим акаунтом (одноразовий вхід у браузері):

       curl -fsSL https://rclone.org/install.sh | sudo bash
       rclone config create gdrive drive scope=drive.readonly root_folder_id=1hjRnP0IO_CxT26AnNiadzR_yGDpOj23u

   Відкриється браузер: увійти акаунтом з доступом до PS VyzAI, Allow, і дочекатись, поки термінал сам повернеться до `$`
   (Ctrl+C на `Waiting for code...` залишає порожній токен; тоді `rclone config delete gdrive` і знову). root_folder_id = тека,
   де лежить `nast_demo`, без нього шлях `gdrive:nast_demo/...` не знайдеться.

Далі все одною командою (скачує 19 ГБ, звіряє md5, запускає інсталер; можна перезапускати, докачає тільки бите):

       rclone copy gdrive:nast_demo/mode3_5090/bootstrap.sh . && bash bootstrap.sh

Після цього одразу крок 4. Кроки 1-3 нижче = те саме руками або з USB.

## Кроки

1. Скопіювати всю папку на ноут, наприклад у `~/mode3_5090` (USB або Drive, як зручніше). Перевірити, що доїхало цілим:

       cd ~/mode3_5090 && md5sum -c manifest.md5

   Усі рядки мають бути `OK`. Файл із `FAILED` копіювати заново.

2. Префлайт (інвентар машини і запису, 10 секунд):

       bash nast_preflight.sh /шлях/до/нового/запису

   Дивитись `nast_preflight_report.txt`: GPU і драйвер, диски, вердикт по місцю, скільки кадрів `A_` і `B_` у записі. Якщо вердикт червоний, спершу звільнити місце або взяти інший диск.

3. Інсталяція (15-30 хвилин, попросить пароль sudo для apt):

       bash install_mode3_clean.sh

   На інший диск: `NAST_ROOT=/data/nast bash install_mode3_clean.sh`. Усе, що робив скрипт, лишається в `install_mode3_clean.log` поруч зі скриптом. Кроки: пакети -> код у `~/nast/nast-local-pipeline` -> venv + torch -> TRELLIS-env у `~/nast/nast_trellis` (звіряє md5 паку, розпаковує, ставить ваги з бандла, робить смоук-генерацію на тестовому кропі) -> старт сервісу на порту 8130 і зупинка. У кінці має бути `done` і рядок `service: online`.

4. Запуск:

       ~/nast/nast-local-pipeline/run.sh

   Або ярлик `NAST Deskview` на робочому столі. У шапці апки має бути `Backend online`. Перший старт відкриє порожню сцену, це нормально.

5. Новий запис. Усі `.raw12` кадри обох камер лежать в одній папці, камера відрізняється префіксом імені (`A_...`, `B_...`), окремих папок не треба. Кнопка `Open data folder` -> та папка. Декод і глибина стартують самі, прогрес у шапці. Дочекатись `depth N/N`.

6. Вкладка MAP -> `poses`. Рахує пози обох камер і глибину в їх геометрії (`depth_geo`), приблизно 25 с на 100 кадрів на 5070 Ti, на 5090 швидше. У тексті готового job-а: `A looks forward, B looks backward` (або навпаки, це визначається з даних; задавати нічого не треба).

7. MAP -> `build map` -> `A + B`. Приблизно 30 с на 100 кадрів. Якщо карта не з'явилась сама, кнопка `↻` біля вибору сцени. `Free orbit`, ЛКМ крутити, коліщатко зум, клік по точці = орбіта навколо неї.

8. Об'єкти: вкладка REC, обвести об'єкт на кадрі (ROI) -> `Auto views` (до 8 ракурсів з обох камер і обох напрямків проїзду) -> `Reconstruct`. Далі SAM + Real-ESRGAN + TRELLIS, 3-5 хвилин на об'єкт. Результат у вкладці 3D: гаусіани, кнопка `Mesh` = меш зі структурними режимами, справа реальні ракурси.

## Новий запис 18 вересня (multimode downtown)

На Drive `PS VyzAI - part 2 / Dash Cam / ... / September Demo Data / 18_Sep_2026_(multimode_downtown)`: два блоки Orthovector, 1484 = перед, 1649 = зад, по три шари архівів на камеру (L1 = 1 fps, L1+L2 = 5 fps, L1+L2+L3 = 15 fps). Для карти й об'єктів беремо L1+L2 = 5 fps (29 ГБ архівів, 4119 кадрів, ~41 ГБ після декоду).

1. Скачати (якщо архівів ще нема в `~/Downloads`; тека вказана своїм ID, бо лежить на shared drive):

       cd ~/Downloads && rclone copy "gdrive,root_folder_id=1Z263dTC1c-ojoZx4iSdo1bIEQMMTmUIL,team_drive=0AFniWOsxQDmoUk9PVA:" . --include "L*.tar" --include "*.md" -P

2. Перевірити списком, не розміром (друге число = першому):

       cd ~/Downloads && for f in L1_1fps_1484_front.tar:429 L2_to5fps_1484_front.tar:1709 L1_1fps_1649_rear.tar:399 L2_to5fps_1649_rear.tar:1588; do n=${f%%:*}; echo "$n $(tar -tf "$n" | grep -vc '/$') / ${f##*:}"; done

3. Розпакувати строго по порядку, L1 потім L2 (очікувано 4119):

       mkdir -p ~/rec_2026_09_18 && cd ~/rec_2026_09_18 && for f in L1_1fps_1484_front L2_to5fps_1484_front L1_1fps_1649_rear L2_to5fps_1649_rear; do tar -xf ~/Downloads/$f.tar || break; done; find . -name '*.raw' | wc -l

4. Зібрати з двох папок одну папку запису: переднє = A, заднє = B (жорсткі посилання, місця не їсть) плюс `rig.json` з орієнтацією кожної камери з її `session.json`:

       ~/nast/nast-local-pipeline/venv/bin/python ~/nast/nast-local-pipeline/monocars/import_rig.py ~/rec_2026_09_18/rig "$HOME/rec_2026_09_18/18_Sep_2026_(multimode_downtown)/1484_front" "$HOME/rec_2026_09_18/18_Sep_2026_(multimode_downtown)/1649_rear"

   Очікувано: A 2135 кадрів, B 1984, `upright = sensor image turned 180 deg` (риг стоїть горизонтально, сенсор перевернутий; декодер кладе кадри повернутими, REC показує рівно), перекриття камер у часі ~396 с.

5. В апці `Open data folder` -> `~/rec_2026_09_18/rig`, дочекатись `depth 4119/4119`, далі MAP -> `poses` і `build map` -> `A + B`.

Кадри 18.09 мають ту саму байтову розкладку, що й 07.08 (2448x2048 RAW12, рядок 3680 байтів). Камери НЕ синхронізовані по кадрах, лише по годиннику (обидва звірені з UTC): пайплайн зв'язує їх за часовими мітками з імен, покадрове парування не потрібне. IMU і GNSS у записі порожні, вони не використовуються.

## Якщо щось пішло не так

- Інсталер: `~/mode3_5090/install_mode3_clean.log`.
- Сервіс: `~/nast/nast-local-pipeline/inspector/srv.log` і `srv.err`.
- TRELLIS на конкретному об'єкті: `~/nast/nast-local-pipeline/inspector/jobs/job_<id>/trellis.log` (SAM score кожного виду, скільки видів узято, етапи генерації, out of memory).
- Смоук TRELLIS після розпакування: `~/nast/nast_trellis/smoke/` (має бути `asset.ply`).
- Пози нового запису: `<папка запису>/map/poses_report.json` (площина дороги по чанках, висота камери, поворот рига, куди дивиться яка камера).
- Перерахунок поз = новий світ: стара карта переїжджає в `<папка запису>/map/stale_<час>/`, об'єкти, розв'язані на старих позах, треба розв'язати заново (job про це попередить).
- TRELLIS-пак не розпакувався: запасний шлях без паку, 40-60 хвилин з компіляцією на місці:

       NAST_TRELLIS_ROOT=~/nast/nast_trellis bash ~/nast/nast-local-pipeline/local_gpu/trellis/install_trellis.sh

- Інший риг (камери не спина до спини, а рознесені): перед `run.sh` задати `NAST_RIG_LEVER=x,y,z` (зсув другої камери відносно опорної в її системі координат, у висотах камери).
