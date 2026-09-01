using System.IO;
using IOPath = System.IO.Path;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace NastDeskview;

public partial class MainWindow : Window
{
    // ---------------- data ----------------
    private string _root;
    private List<string> _framesA = new(), _framesB = new();
    private string _cam = "A";
    private string _layer = "rgb";
    private string _variant = "nxyz";              // folder (relative to root) behind "Normals" / "ROI normals"
    private sealed class VariantEntry { public string Name = "", Dir = ""; public override string ToString() => Name; }
    // locked catalog, client naming (2026-08-27): no scalar polarization views,
    // no algorithm-author names — just phys / diffuse / specv2
    private static readonly Dictionary<string, string> VariantNames = new()
    {
        ["nxyz"] = "Nxyz", ["n_xy"] = "N xy", ["n_xz"] = "N xz",
        ["nxyz_phys"] = "phys", ["nxyz_diffuse"] = "diffuse", ["nxyz_specv2"] = "specv2",
        ["edge"] = "Edge", ["rgb_deglare"] = "RGB deglare",
    };
    private static readonly string[] VariantOrder =
    {
        "nxyz", "n_xy", "n_xz", "nxyz_phys", "nxyz_diffuse", "nxyz_specv2", "edge", "rgb_deglare",
    };
    private int _pos;
    private bool _playing;
    private int _dir = 1;
    private double _speed = 1, _acc;
    private const double Fps = 8.0;
    private const double UnitsToMeters = 3.38;      // COLMAP unit ≈ 3.38 m on this rig

    private readonly Dictionary<string, BitmapSource> _cache = new();
    private readonly LinkedList<string> _lru = new();
    private const int CacheMax = 240;

    private readonly Api _api = new();
    private MetaDto _meta;
    private Dictionary<string, FramePose> _poses;
    private List<ObjDto> _objects = new();
    private int _sel = -1;
    private int _addTo = -1;

    private string _tool;
    private List<Point> _drawPts;
    private bool _dragging;

    private readonly DispatcherTimer _tick = new() { Interval = TimeSpan.FromMilliseconds(33) };
    private readonly DispatcherTimer _jobs = new() { Interval = TimeSpan.FromSeconds(3) };
    private readonly DispatcherTimer _toastHide = new() { Interval = TimeSpan.FromSeconds(2.6) };
    private DateTime _last = DateTime.Now;
    private bool _squelch;
    private readonly List<Button> _speedBtns = new();

    private Brush B(string key) => (Brush)FindResource(key);

    private void Toast(string msg, bool bad = false)
    {
        ToastText.Text = msg;
        ToastText.Foreground = bad ? B("ErrLight") : B("Fg");
        ToastBox.BorderBrush = bad
            ? new SolidColorBrush(Color.FromArgb(0x66, 0xFF, 0x6B, 0x5E))
            : new SolidColorBrush(Color.FromArgb(0x66, 0x55, 0xDC, 0x78));
        ToastBox.Visibility = Visibility.Visible;
        _toastHide.Stop(); _toastHide.Start();
    }

    public MainWindow()
    {
        InitializeComponent();
        foreach (var s in new[] { 0.25, 0.5, 1, 2, 4, 8 })
        {
            var b = new Button
            {
                Content = s + "×",
                FontFamily = (FontFamily)FindResource("Mono"),
                FontSize = 10.5,
                Padding = new Thickness(8, 5, 8, 5),
                Margin = new Thickness(1, 0, 1, 0),
                BorderThickness = new Thickness(0),
                Background = Brushes.Transparent,
                Foreground = B("Faint"),
                Tag = s,
            };
            b.Click += (_, _) => SetSpeed((double)b.Tag);
            _speedBtns.Add(b);
            SpeedPanel.Children.Add(b);
        }
        _tick.Tick += Tick;
        _jobs.Tick += async (_, _) => await PollJobs();
        _toastHide.Tick += (_, _) => { ToastBox.Visibility = Visibility.Collapsed; _toastHide.Stop(); };
        Loaded += async (_, _) =>
        {
            var def = @"G:\nast_mode3\viewer\scenes\street_video";
            if (Directory.Exists(IOPath.Combine(def, "rgb"))) LoadFolder(def);
            await EnsureServer();
            await ConnectApi();
            _tick.Start();
            _jobs.Start();
            SetSpeed(1);
            HiliteButtons();
            SwitchTab(0);
        };
    }

    private void SetSpeed(double s)
    {
        _speed = s;
        foreach (var b in _speedBtns)
        {
            bool on = Math.Abs((double)b.Tag - s) < 1e-9;
            b.Background = on ? B("Raised2") : Brushes.Transparent;
            b.Foreground = on ? B("Fg") : B("Faint");
            b.FontWeight = on ? FontWeights.SemiBold : FontWeights.Medium;
        }
    }

    // ---------------- tabs ----------------
    private void ShowRec(object s, RoutedEventArgs e) => SwitchTab(0);
    private async void ShowMap(object s, RoutedEventArgs e)
    {
        SwitchTab(1);
        if (Web.Source == null) await LoadScenes();
    }
    private void Show3d(object s, RoutedEventArgs e)
    {
        SwitchTab(2);
        LoadRenders();
    }

    private void SwitchTab(int i)
    {
        PaneRec.Visibility = i == 0 ? Visibility.Visible : Visibility.Collapsed;
        PaneMap.Visibility = i == 1 ? Visibility.Visible : Visibility.Collapsed;
        Pane3d.Visibility = i == 2 ? Visibility.Visible : Visibility.Collapsed;
        var tabs = new[] { (TabRecBtn, RailRecLbl), (TabMapBtn, RailMapLbl), (Tab3dBtn, Rail3dLbl) };
        for (int k = 0; k < 3; k++)
        {
            var (btn, lbl) = tabs[k];
            bool on = k == i;
            btn.Background = on ? B("Raised") : Brushes.Transparent;
            lbl.Foreground = on ? B("Accent") : B("Faint");
            if (btn.Content is StackPanel sp && sp.Children[0] is Border icon)
                icon.BorderBrush = on ? B("Accent") : B("Faint");
        }
    }

    // ---------------- server lifecycle ----------------
    private async Task EnsureServer()
    {
        if ((await _api.Meta()) != null) return;
        try
        {
            // portable: prefer the inspector dir shipped next to the app
            // (<exe>\..\inspector, the demo-kit layout), fall back to the dev path;
            // a venv python next to the kit wins over the PATH one
            string baseDir = System.AppContext.BaseDirectory;
            string insp = System.IO.Path.GetFullPath(System.IO.Path.Combine(baseDir, "..", "inspector"));
            if (!System.IO.Directory.Exists(insp)) insp = @"G:\nast_mode3\inspector";
            string venvPy = System.IO.Path.GetFullPath(System.IO.Path.Combine(insp, "..", "venv", "Scripts", "python.exe"));
            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = System.IO.File.Exists(venvPy) ? venvPy : "python",
                Arguments = "server.py 8130",
                WorkingDirectory = insp,
                CreateNoWindow = true,
                UseShellExecute = false,
            };
            System.Diagnostics.Process.Start(psi);
            for (int i = 0; i < 20; i++)
            {
                await Task.Delay(700);
                if ((await _api.Meta()) != null) return;
            }
        }
        catch { }
    }

    private async Task ConnectApi()
    {
        _meta = await _api.Meta();
        if (_meta != null)
        {
            LblApi.Text = $"Backend online · {(_meta.count / 1e6):F1}M pts indexed";
            ApiDot.Fill = B("Go");
            try { _poses = await _api.Poses(); } catch { _poses = null; }
            await RefreshObjects();
        }
        else
        {
            LblApi.Text = "Backend offline — playback only";
            ApiDot.Fill = B("Err");
        }
    }

    // ---------------- folder / playback ----------------
    private void OpenFolder(object s, RoutedEventArgs e)
    {
        var dlg = new Microsoft.Win32.OpenFolderDialog { Title = "Folder with rgb/ and nxyz/" };
        if (dlg.ShowDialog() == true) LoadFolder(dlg.FolderName);
    }

    private void LoadFolder(string root)
    {
        var rgb = IOPath.Combine(root, "rgb");
        if (!Directory.Exists(rgb))
        {
            MessageBox.Show("The folder has no rgb/ subfolder"); return;
        }
        _root = root;
        var all = Directory.GetFiles(rgb).Select(IOPath.GetFileName).OrderBy(x => x).ToList();
        _framesA = all.Where(f => f.StartsWith("A_")).ToList();
        _framesB = all.Where(f => f.StartsWith("B_")).ToList();
        if (_framesA.Count == 0) _framesA = all;
        _cache.Clear(); _lru.Clear();
        _pos = 0;
        LblDataset.Text = IOPath.GetFileName(root);
        LblDatasetPath.Text = IOPath.GetDirectoryName(root);
        LblFrames.Text = $"A {_framesA.Count}   B {_framesB.Count}";
        LoadVariants();
        Render();
    }

    // the polarization products available for this dataset: nxyz/ from the
    // recorder plus every folder under layers/ (polar_layers.py output)
    private void LoadVariants()
    {
        var found = new Dictionary<string, string>();
        if (Directory.Exists(IOPath.Combine(_root, "nxyz"))) found["nxyz"] = "nxyz";
        var ld = IOPath.Combine(_root, "layers");
        if (Directory.Exists(ld))
            foreach (var d in Directory.GetDirectories(ld))
                found[IOPath.GetFileName(d)] = "layers/" + IOPath.GetFileName(d);
        CmbVariant.Items.Clear();
        foreach (var k in VariantOrder.Concat(found.Keys.Except(VariantOrder)))
            if (found.TryGetValue(k, out var dir))
                CmbVariant.Items.Add(new VariantEntry { Name = VariantNames.TryGetValue(k, out var n) ? n : k, Dir = dir });
        CmbVariant.Visibility = CmbVariant.Items.Count > 1 ? Visibility.Visible : Visibility.Collapsed;
        var cur = CmbVariant.Items.Cast<VariantEntry>().FirstOrDefault(v => v.Dir == _variant)
                  ?? CmbVariant.Items.Cast<VariantEntry>().FirstOrDefault();
        _squelch = true; CmbVariant.SelectedItem = cur; _squelch = false;
        if (cur != null) _variant = cur.Dir;
    }

    private void VariantChanged(object s, SelectionChangedEventArgs e)
    {
        if (_squelch || CmbVariant.SelectedItem is not VariantEntry v) return;
        _variant = v.Dir;
        if (_layer == "rgb") _layer = "nxyz";        // picking a product means "show it"
        HiliteButtons(); Render();
    }

    private List<string> Frames => _cam == "A" ? _framesA : _framesB;
    private string CurName => Frames.Count == 0 ? null : Frames[Math.Clamp(_pos, 0, Frames.Count - 1)];

    private void Tick(object s, EventArgs e)
    {
        var now = DateTime.Now;
        var dt = (now - _last).TotalSeconds; _last = now;
        if (_playing && Frames.Count > 0)
        {
            _acc += dt * Fps * _speed;
            int st = (int)_acc;
            if (st > 0)
            {
                _acc -= st;
                _pos += st * _dir;
                if (_pos <= 0) { _pos = 0; if (_dir < 0) Pause(); }
                if (_pos >= Frames.Count - 1) { _pos = Frames.Count - 1; if (_dir > 0) Pause(); }
                Render();
            }
        }
    }

    private void Pause() { _playing = false; BtnPlay.Content = "▶"; }

    private void TogglePlay(object s, RoutedEventArgs e)
    {
        _playing = !_playing;
        BtnPlay.Content = _playing ? "⏸" : "▶";
        _acc = 0; _last = DateTime.Now;
    }

    private void ToggleReverse(object s, RoutedEventArgs e)
    {
        _dir = -_dir;
        BtnRev.Background = _dir < 0 ? B("Raised2") : B("Panel2");
        BtnRev.Foreground = _dir < 0 ? B("Fg") : B("Dim");
    }

    private void StepBack(object s, RoutedEventArgs e) { _pos = Math.Max(0, _pos - 1); Render(); }
    private void StepFwd(object s, RoutedEventArgs e) { _pos = Math.Min(Frames.Count - 1, _pos + 1); Render(); }

    private void TimelineChanged(object s, RoutedPropertyChangedEventArgs<double> e)
    {
        if (_squelch) return;
        _pos = (int)e.NewValue; Render();
    }

    private void CamA(object s, RoutedEventArgs e) { _cam = "A"; _pos = Math.Min(_pos, _framesA.Count - 1); HiliteButtons(); Render(); }
    private void CamB(object s, RoutedEventArgs e) { if (_framesB.Count == 0) return; _cam = "B"; _pos = Math.Min(_pos, _framesB.Count - 1); HiliteButtons(); Render(); }
    private void LayerRgb(object s, RoutedEventArgs e) { _layer = "rgb"; HiliteButtons(); Render(); }
    private void LayerNx(object s, RoutedEventArgs e) { _layer = "nxyz"; HiliteButtons(); Render(); }
    private void LayerRoiNx(object s, RoutedEventArgs e) { _layer = "roinx"; HiliteButtons(); Render(); }
    private void ToolRect(object s, RoutedEventArgs e) { _tool = _tool == "rect" ? null : "rect"; HiliteButtons(); }
    private void ToolPoly(object s, RoutedEventArgs e) { _tool = _tool == "poly" ? null : "poly"; _drawPts = null; HiliteButtons(); }

    private void HiliteButtons()
    {
        void Seg(Button b, bool on, bool amber = false)
        {
            if (on && amber) { b.Background = B("Accent"); b.Foreground = B("OnAccent"); b.FontWeight = FontWeights.SemiBold; }
            else if (on) { b.Background = B("Raised2"); b.Foreground = B("Fg"); b.FontWeight = FontWeights.SemiBold; }
            else { b.Background = Brushes.Transparent; b.Foreground = B("Dim"); b.FontWeight = FontWeights.Medium; }
        }
        Seg(BtnCamA, _cam == "A", amber: true);
        Seg(BtnCamB, _cam == "B", amber: true);
        Seg(BtnRgb, _layer == "rgb");
        Seg(BtnNx, _layer == "nxyz");
        Seg(BtnRoiNx, _layer == "roinx");

        void Draw(Button b, bool on)
        {
            if (on)
            {
                b.Background = new SolidColorBrush(Color.FromArgb(0x1F, 0x55, 0xDC, 0x78));
                b.BorderBrush = new SolidColorBrush(Color.FromArgb(0x73, 0x55, 0xDC, 0x78));
                b.Foreground = B("GoLight");
                b.FontWeight = FontWeights.SemiBold;
            }
            else
            {
                b.Background = B("Panel2");
                b.BorderBrush = B("Line3");
                b.Foreground = B("Dim");
                b.FontWeight = FontWeights.Medium;
            }
        }
        Draw(BtnRect, _tool == "rect");
        Draw(BtnPoly, _tool == "poly");
    }

    // ---------------- frames ----------------
    private BitmapSource Frame(string name, string layer)
    {
        var key = layer + "/" + name;
        if (layer.StartsWith("layers/") && !File.Exists(IOPath.Combine(_root, layer, name)))
            return Frame(name, "nxyz");                 // product not generated for this frame (yet)
        if (_cache.TryGetValue(key, out var hit))
        {
            _lru.Remove(key); _lru.AddLast(key);
            return hit;
        }
        var path = IOPath.Combine(_root, layer, name);
        if (!File.Exists(path)) return null;
        var bi = new BitmapImage();
        bi.BeginInit();
        bi.CacheOption = BitmapCacheOption.OnLoad;
        bi.UriSource = new Uri(path);
        bi.EndInit();
        var up = new TransformedBitmap(bi, new RotateTransform(90));
        up.Freeze();
        _cache[key] = up; _lru.AddLast(key);
        if (_lru.Count > CacheMax)
        {
            _cache.Remove(_lru.First.Value); _lru.RemoveFirst();
        }
        return up;
    }

    private void Render()
    {
        var name = CurName;
        if (name == null) return;
        var baseLayer = _layer == "nxyz" ? _variant : "rgb";
        var im = Frame(name, baseLayer);
        if (im == null) return;
        if (Math.Abs(Stage.Width - im.PixelWidth) > 0.5)
        {
            Stage.Width = im.PixelWidth; Stage.Height = im.PixelHeight;
        }
        ImgMain.Source = im;

        if (_layer == "roinx")
        {
            var o = _objects.FirstOrDefault(x => x.id == _sel);
            var geo = o != null ? RoiGeometry(o) : null;
            if (geo != null)
            {
                ImgPunch.Source = Frame(name, _variant);
                ImgPunch.Clip = geo;
                ImgPunch.Visibility = Visibility.Visible;
            }
            else ImgPunch.Visibility = Visibility.Collapsed;
        }
        else ImgPunch.Visibility = Visibility.Collapsed;

        _squelch = true;
        Timeline.Maximum = Math.Max(1, Frames.Count - 1);
        Timeline.Value = _pos;
        _squelch = false;
        LblPosCur.Text = _pos.ToString();
        LblPosMax.Text = $" / {Frames.Count - 1}";

        // glass HUD
        HudFrame.Text = $"{_cam}_{_pos:D6}";
        int nRoi = _objects.Count(o => o.frame == name);
        HudRoiChip.Visibility = nRoi > 0 ? Visibility.Visible : Visibility.Collapsed;
        HudRoiText.Text = nRoi == 1 ? "1 ROI on this frame" : $"{nRoi} ROIs on this frame";

        DrawOverlay(name);
    }

    private Geometry ViewGeometry(ObjDto.View v)
    {
        try
        {
            var pts = v.pts.Select(p => new Point(p[0], p[1])).ToList();
            if (v.kind == "rect" && pts.Count >= 2)
                return new RectangleGeometry(new Rect(pts[0], pts[1]));
            if (pts.Count >= 3)
            {
                var fig = new PathFigure { StartPoint = pts[0], IsClosed = true };
                foreach (var p in pts.Skip(1)) fig.Segments.Add(new LineSegment(p, true));
                return new PathGeometry(new[] { fig });
            }
        }
        catch { }
        return null;
    }

    private Geometry RoiGeometry(ObjDto o)
    {
        try
        {
            var pts = JsonSerializer.Deserialize<double[][]>(o.pts);
            if (o.kind == "rect" && pts.Length >= 2)
            {
                var r = new Rect(new Point(Math.Min(pts[0][0], pts[1][0]), Math.Min(pts[0][1], pts[1][1])),
                                 new Point(Math.Max(pts[0][0], pts[1][0]), Math.Max(pts[0][1], pts[1][1])));
                return new RectangleGeometry(r);
            }
            var g = new StreamGeometry();
            using (var c = g.Open())
            {
                c.BeginFigure(new Point(pts[0][0], pts[0][1]), true, true);
                c.PolyLineTo(pts.Skip(1).Select(p => new Point(p[0], p[1])).ToList(), true, true);
            }
            return g;
        }
        catch { return null; }
    }

    private void DrawOverlay(string name)
    {
        Overlay.Children.Clear();
        foreach (var o in _objects)
        {
            foreach (var v in o.Views)
            {
                if (v.frame != name) continue;
                var geo = ViewGeometry(v);
                if (geo == null) continue;
                bool primary = v.frame == o.frame && !v.auto;
                Overlay.Children.Add(new System.Windows.Shapes.Path
                {
                    Data = geo,
                    Stroke = v.auto ? B("Accent") : B("Go"),
                    StrokeThickness = primary ? 3 : 2,
                    StrokeDashArray = v.auto ? new DoubleCollection { 5, 4 } : null,
                    Fill = new SolidColorBrush(v.auto ? Color.FromArgb(14, 0xF2, 0xA9, 0x3B) : Color.FromArgb(20, 0x55, 0xDC, 0x78)),
                    ToolTip = v.auto ? $"{o.label} · auto view" : $"{o.label}",
                });
                if (v.auto)
                {
                    var tag = new TextBlock
                    {
                        Text = $"{o.label} · auto", FontSize = 20, Foreground = B("Accent"),   // overlay is scaled with the frame
                        FontFamily = (FontFamily)FindResource("Mono"),
                    };
                    var b = geo.Bounds;
                    Canvas.SetLeft(tag, b.Left + 4); Canvas.SetTop(tag, Math.Max(0, b.Top - 16));
                    Overlay.Children.Add(tag);
                }
            }
        }
        if (_drawPts is { Count: > 0 })
        {
            var pl = new Polyline
            {
                Stroke = B("Accent"), StrokeThickness = 2.5,
                StrokeDashArray = new DoubleCollection { 6, 4 },
            };
            foreach (var p in _drawPts) pl.Points.Add(p);
            Overlay.Children.Add(pl);
        }
    }

    // ---------------- ROI drawing ----------------
    private void OverlayDown(object s, MouseButtonEventArgs e)
    {
        if (_tool == null) return;
        var p = e.GetPosition(Overlay);
        if (_tool == "poly")
        {
            _drawPts ??= new List<Point>();
            if (e.ClickCount == 2)
            {
                if (_drawPts.Count >= 3) _ = CommitRoi("poly", _drawPts.ToList());
                _drawPts = null; Render(); return;
            }
            _drawPts.Add(p); Render();
        }
        else
        {
            _drawPts = new List<Point> { p, p };
            _dragging = true;
            Overlay.CaptureMouse();
        }
    }

    private void OverlayMove(object s, MouseEventArgs e)
    {
        if (!_dragging || _drawPts == null) return;
        _drawPts[1] = e.GetPosition(Overlay);
        DrawRectPreview();
    }

    private void DrawRectPreview()
    {
        Render();
        var a = _drawPts[0]; var b = _drawPts[1];
        Overlay.Children.Add(new Rectangle
        {
            Width = Math.Abs(b.X - a.X), Height = Math.Abs(b.Y - a.Y),
            Stroke = B("Accent"), StrokeThickness = 2.5,
            StrokeDashArray = new DoubleCollection { 6, 4 },
            RenderTransform = new TranslateTransform(Math.Min(a.X, b.X), Math.Min(a.Y, b.Y)),
        });
    }

    private void OverlayUp(object s, MouseButtonEventArgs e)
    {
        if (!_dragging) return;
        _dragging = false;
        Overlay.ReleaseMouseCapture();
        var a = _drawPts[0]; var b = _drawPts[1];
        _drawPts = null;
        if (Math.Abs(b.X - a.X) < 6 || Math.Abs(b.Y - a.Y) < 6) { Render(); return; }
        _ = CommitRoi("rect", new List<Point> { a, b });
    }

    private async Task CommitRoi(string kind, List<Point> pts)
    {
        try { await CommitRoiInner(kind, pts); }
        catch (Exception ex) { Toast("ROI: " + ex.Message, true); }
    }

    private async Task CommitRoiInner(string kind, List<Point> pts)
    {
        if (!_api.Alive) { Toast("Backend offline — ROI can't be solved", true); return; }
        var rp = pts.Select(p => new[] { (int)p.X, (int)p.Y }).ToArray();
        if (_addTo > 0)
        {
            var id = _addTo; _addTo = -1;
            BtnAddObs.Background = B("Panel2"); BtnAddObs.Foreground = B("Fg2");
            Toast("Merging the extra view…");
            var (ok, body) = await _api.Post($"/api/objects/{id}/roi",
                new { frame = CurName, kind, pts = rp });
            if (ok)
                Toast($"✓ Box refit: {body.GetProperty("npts").GetInt32():N0} pts, " +
                      $"{JsonDocument.Parse(body.GetProperty("obs").GetString()).RootElement.GetArrayLength()} views");
            else Toast("View failed: " + body, true);
            await RefreshObjects();
            return;
        }
        var label = string.IsNullOrWhiteSpace(TxtLabel.Text) ? $"R{_objects.Count + 1}" : TxtLabel.Text.Trim();
        Toast("Solving 3D position…");
        var (ok2, body2) = await _api.Post("/api/roi",
            new { frame = CurName, cam = _cam, kind, pts = rp, label });
        if (ok2)
            Toast($"✓ 3D position solved — {body2.GetProperty("npts").GetInt32():N0} points");
        else
            Toast("ROI failed: " + body2, true);
        await RefreshObjects();
        if (ok2)
        {
            _sel = body2.GetProperty("id").GetInt32(); RebuildObjList(); Select(_objects.FirstOrDefault(x => x.id == _sel), keep: true); Render();
            // collect more views of the new object by itself (poses + depth + template check)
            try
            {
                var k = await RunAutoViews(_sel);
                if (k > 0) Toast($"✓ 3D solved · {k} auto view{(k == 1 ? "" : "s")} collected");
                await RefreshObjects();
                Select(_objects.FirstOrDefault(x => x.id == _sel), keep: true);
            }
            catch { }
        }
    }

    // ---------------- objects panel ----------------
    private async Task RefreshObjects()
    {
        try { _objects = await _api.Objects() ?? new List<ObjDto>(); }
        catch { _objects = new List<ObjDto>(); }
        RebuildObjList();
        Render();
    }

    private void RebuildObjList()
    {
        ObjList.Items.Clear();
        LblObjCount.Text = _objects.Count.ToString();
        foreach (var o in _objects)
        {
            bool sel = o.id == _sel;
            var bar = new Border
            {
                Width = 3, Height = 26, CornerRadius = new CornerRadius(2),
                Background = sel ? B("Accent") : B("Line3"),
                VerticalAlignment = VerticalAlignment.Center,
            };
            var title = new TextBlock { Text = o.label, FontWeight = FontWeights.SemiBold, FontSize = 12,
                Foreground = sel ? B("Fg") : B("Fg2") };
            var meta1 = new TextBlock
            {
                Text = $"{o.kind?.ToUpper()} · CAM {o.cam}",
                FontFamily = (FontFamily)FindResource("Mono"), FontSize = 9.5,
                Foreground = sel ? B("GoLight") : B("Faint"),
                Margin = new Thickness(8, 1, 0, 0),
            };
            var meta2 = new TextBlock
            {
                Text = $"{o.npts:N0} pts · {o.ObsCount} view{(o.ObsCount == 1 ? "" : "s")}" +
                       (o.AutoCount > 0 ? $" ({o.AutoCount} auto)" : ""),
                FontFamily = (FontFamily)FindResource("Mono"), FontSize = 10.5,
                Foreground = sel ? B("Dim") : B("Faint"),
            };
            var col = new StackPanel { Margin = new Thickness(10, 0, 8, 0), VerticalAlignment = VerticalAlignment.Center };
            var row1 = new StackPanel { Orientation = Orientation.Horizontal };
            row1.Children.Add(title); row1.Children.Add(meta1);
            col.Children.Add(row1); col.Children.Add(meta2);

            var jump = new Button
            {
                Content = "→ frame", FontFamily = (FontFamily)FindResource("Mono"), FontSize = 10,
                Padding = new Thickness(8, 5, 8, 5), VerticalAlignment = VerticalAlignment.Center,
            };
            jump.Click += (_, ev) => { ev.Handled = true; JumpTo(o); };
            var del = new Button
            {
                Content = "✕", Width = 24, Height = 24, Padding = new Thickness(0),
                VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(4, 2, 0, 2),
            };
            del.Click += async (_, ev) =>
            {
                ev.Handled = true;
                await _api.Delete($"/api/objects/{o.id}");
                if (_sel == o.id) { _sel = -1; PosePanel.Visibility = Visibility.Collapsed; }
                await RefreshObjects();
            };

            var dock = new DockPanel { Margin = new Thickness(0) };
            DockPanel.SetDock(del, Dock.Right);
            DockPanel.SetDock(jump, Dock.Right);
            dock.Children.Add(del); dock.Children.Add(jump);
            var left = new StackPanel { Orientation = Orientation.Horizontal };
            left.Children.Add(bar); left.Children.Add(col);
            dock.Children.Add(left);

            var card = new Border
            {
                Background = sel ? B("Raised") : B("Card"),
                BorderBrush = sel ? new SolidColorBrush(Color.FromArgb(0x66, 0xF2, 0xA9, 0x3B)) : B("Line"),
                BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(8),
                Padding = new Thickness(10, 9, 10, 9), Margin = new Thickness(0, 0, 0, 5),
                Child = dock, Cursor = Cursors.Hand,
            };
            card.MouseLeftButtonUp += (_, _) => Select(o);
            ObjList.Items.Add(card);
        }
    }

    private void Select(ObjDto o, bool keep = false)
    {
        if (o == null) { PosePanel.Visibility = Visibility.Collapsed; return; }
        _sel = (!keep && _sel == o.id) ? -1 : o.id;
        var cur = _objects.FirstOrDefault(x => x.id == _sel);
        if (cur?.P?.t != null)
        {
            PosePanel.Visibility = Visibility.Visible;
            PoseTitle.Text = cur.label;
            var sz = cur.P.size;
            LblSelSize.Text = $"{sz[0] * UnitsToMeters:F1} × {sz[2] * UnitsToMeters:F1} × {sz[1] * UnitsToMeters:F1} m";
            LblSelPts.Text = $"{cur.npts:N0}";
            var nv = cur.Views.Count; var na = cur.AutoCount;
            LblViews.Text = nv <= 1 && na == 0
                ? "1 view — add more or run Auto views for a better reconstruction"
                : $"{nv} views for the reconstruction · {nv - na} yours, {na} auto (dashed amber on the frames)";
        }
        else PosePanel.Visibility = Visibility.Collapsed;
        RebuildObjList();
        Render();
    }

    private void JumpTo(ObjDto o)
    {
        if (o.cam == "B" && _framesB.Count > 0) _cam = "B"; else _cam = "A";
        var i = Frames.IndexOf(o.frame);
        if (i >= 0) _pos = i;
        HiliteButtons();
        Render();
    }

    private void AddObs(object s, RoutedEventArgs e)
    {
        if (_sel < 0) return;
        _addTo = _addTo > 0 ? -1 : _sel;
        if (_addTo > 0)
        {
            BtnAddObs.Background = B("Raised2"); BtnAddObs.Foreground = B("Fg");
            if (_tool == null) { _tool = "rect"; HiliteButtons(); }
            Toast("Go to another frame and outline the same object");
        }
        else { BtnAddObs.Background = B("Panel2"); BtnAddObs.Foreground = B("Fg2"); }
    }

    private async Task<int> RunAutoViews(int id, int n = 5)
    {
        var (ok, body) = await _api.Post($"/api/objects/{id}/autoviews", new { n });
        if (!ok) { Toast("Auto views failed: " + body, true); return -1; }
        return body.TryGetProperty("added", out var a) ? a.GetArrayLength() : 0;
    }

    private async void AutoViews(object s, RoutedEventArgs e)
    {
        if (_sel < 0) return;
        Toast("Looking for the object in other frames…");
        try
        {
            var k = await RunAutoViews(_sel);
            if (k >= 0) Toast(k > 0 ? $"✓ {k} auto view{(k == 1 ? "" : "s")} added" : "No other frame shows it well enough", k == 0);
            await RefreshObjects();
            Select(_objects.FirstOrDefault(x => x.id == _sel), keep: true);
        }
        catch (Exception ex) { Toast(ex.Message, true); }
    }

    private async void ExportAugment(object s, RoutedEventArgs e)
    {
        if (_sel < 0) return;
        Toast("Exporting augmentation set… (real crops + synthetic renders)");
        try
        {
            var (ok, body) = await _api.Post($"/api/objects/{_sel}/augment", new { yaw_step = 15, pitches = new[] { 5, 20 }, size = 512 });
            if (!ok) { Toast("Export failed: " + body, true); return; }
            var path = body.GetProperty("path").GetString();
            Toast($"✓ {body.GetProperty("real_views").GetInt32()} real views · {body.GetProperty("synthetic").GetInt32()} synthetic renders · {body.GetProperty("sets").GetArrayLength()} layers → {path}");
            try { System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo("explorer.exe", path) { UseShellExecute = true }); } catch { }
        }
        catch (Exception ex) { Toast(ex.Message, true); }
    }

    private async void Reconstruct(object s, RoutedEventArgs e)
    {
        if (_sel < 0) return;
        try
        {
            var (ok, body) = await _api.Post("/api/reconstruct", new { object_id = _sel });
            Toast(ok ? $"Reconstruction queued — job #{body.GetProperty("job_id").GetInt32()}" : "Failed: " + body, !ok);
        }
        catch (Exception ex) { Toast(ex.Message, true); }
    }

    private async void BuildMap(object s, RoutedEventArgs e)
    {
        var cams = (string)((Button)s).Tag;
        try
        {
            var (ok, body) = await _api.Post("/api/build_map", new { cams });
            Toast(ok
                ? $"Point cloud {cams} — job #{body.GetProperty("job_id").GetInt32()} started"
                : "Map build failed: " + body, !ok);
        }
        catch (Exception ex) { Toast("Map: " + ex.Message, true); }
    }

    // ---------------- jobs panel ----------------
    private async Task PollJobs()
    {
        if (!_api.Alive) return;
        try
        {
            var jobs = await _api.Jobs();
            int running = jobs.Count(j => j.status == "running");
            int queued = jobs.Count(j => j.status == "queued");
            LblJobsSummary.Text = (running > 0 ? $"● {running} RUNNING  " : "") +
                                  (queued > 0 ? $"○ {queued} QUEUED" : "");
            JobsList.Items.Clear();
            var byObj = _objects.ToDictionary(o => o.id, o => o.label);
            foreach (var j in jobs.Take(6))
            {
                string title = j.kind == "map"
                    ? "Point cloud"
                    : "Reconstruction" + (byObj.TryGetValue(j.object_id, out var lb) ? $" · {lb}" : "");
                JobsList.Items.Add(JobCard(j, title));
            }
        }
        catch { }
    }

    private Border JobCard(JobDto j, string title)
    {
        var head = new DockPanel();
        var num = new TextBlock
        {
            Text = $"#{j.id}", FontFamily = (FontFamily)FindResource("Mono"), FontSize = 10,
            Foreground = B("Faint"), VerticalAlignment = VerticalAlignment.Center,
        };
        var chipText = j.status switch
        {
            "running" => "RUNNING", "queued" => "QUEUED",
            "done" => "✓ DONE", "error" => "ERROR", _ => j.status.ToUpper(),
        };
        var chipBg = j.status switch
        {
            "running" => new SolidColorBrush(Color.FromArgb(0x24, 0xF2, 0xA9, 0x3B)),
            "done" => new SolidColorBrush(Color.FromArgb(0x1F, 0x55, 0xDC, 0x78)),
            "error" => new SolidColorBrush(Color.FromArgb(0x24, 0xFF, 0x6B, 0x5E)),
            _ => B("Raised"),
        };
        var chipFg = j.status switch
        {
            "running" => B("Accent"), "done" => B("GoLight"),
            "error" => B("Err"), _ => B("Dim"),
        };
        var chip = new Border
        {
            Background = chipBg, CornerRadius = new CornerRadius(5),
            Padding = new Thickness(7, 3, 7, 3), VerticalAlignment = VerticalAlignment.Center,
            Child = new TextBlock
            {
                Text = chipText, FontFamily = (FontFamily)FindResource("Mono"),
                FontSize = 9.5, FontWeight = FontWeights.SemiBold, Foreground = chipFg,
            },
        };
        DockPanel.SetDock(chip, Dock.Right);
        head.Children.Add(chip);
        var t = new TextBlock
        {
            Text = title, FontSize = 11.5, Margin = new Thickness(8, 0, 8, 0),
            FontWeight = j.status is "running" or "error" ? FontWeights.SemiBold : FontWeights.Medium,
            Foreground = j.status is "running" or "error" ? B("Fg") : B("Fg2"),
            VerticalAlignment = VerticalAlignment.Center, TextTrimming = TextTrimming.CharacterEllipsis,
        };
        head.Children.Add(num);
        head.Children.Add(t);

        var body = new StackPanel();
        body.Children.Add(head);
        if (j.status is "running" or "error")
        {
            body.Children.Add(new TextBlock
            {
                Text = j.detail, FontSize = 10.5, TextWrapping = TextWrapping.Wrap,
                Foreground = j.status == "error" ? new SolidColorBrush(Color.FromRgb(0xD6, 0xA4, 0x9E)) : B("Fg2"),
                Margin = new Thickness(0, 7, 0, 0), MaxHeight = 52,
            });
        }

        return new Border
        {
            Background = j.status == "error" ? new SolidColorBrush(Color.FromRgb(0x15, 0x10, 0x0F)) : B("CardAlt"),
            BorderBrush = j.status switch
            {
                "running" => new SolidColorBrush(Color.FromArgb(0x59, 0xF2, 0xA9, 0x3B)),
                "error" => new SolidColorBrush(Color.FromArgb(0x59, 0xFF, 0x6B, 0x5E)),
                _ => B("Line"),
            },
            BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(9),
            Padding = new Thickness(11), Margin = new Thickness(0, 0, 0, 7),
            Child = body,
        };
    }

    // ---------------- map tab ----------------
    private sealed class SceneEntry
    {
        public string Name = "", Url = "";
        public override string ToString() => Name;
    }

    private async Task LoadScenes()
    {
        try
        {
            var body = await _api.GetJson("/api/scenes");
            var urls = new Dictionary<string, string>();
            foreach (var sc in body.EnumerateArray())
                urls[sc.GetProperty("name").GetString()] = sc.GetProperty("url").GetString();
            CmbScene.Items.Clear();
            if (urls.TryGetValue("street", out var streetUrl))
                CmbScene.Items.Add(new SceneEntry { Name = "Map — point cloud", Url = streetUrl });
            // map with a generated object dropped in: newest finished job per live object
            var jobs = await _api.Jobs();
            var live = _objects.ToDictionary(o => o.id, o => o.label);
            var placed = jobs.Where(j => j.status == "done" && j.kind != "map"
                                         && live.ContainsKey(j.object_id) && urls.ContainsKey($"job_{j.id}"))
                             .GroupBy(j => j.object_id).Select(g => g.OrderByDescending(j => j.id).First())
                             .OrderByDescending(j => j.id);
            foreach (var j in placed)
                CmbScene.Items.Add(new SceneEntry { Name = $"Map + {live[j.object_id]} (job #{j.id})", Url = urls[$"job_{j.id}"] });
            CmbScene.SelectedIndex = CmbScene.Items.Count > 0 ? 0 : -1;
        }
        catch { }
    }

    private async void ReloadScenes(object s, RoutedEventArgs e) => await LoadScenes();

    private async void SceneChanged(object s, SelectionChangedEventArgs e)
    {
        if (CmbScene.SelectedItem == null) return;
        try
        {
            await Web.EnsureCoreWebView2Async();
            Web.Source = new Uri(_api.Base + ((SceneEntry)CmbScene.SelectedItem).Url + "?top=1");
        }
        catch (Exception ex) { LblApi.Text = "WebView2: " + ex.Message; }
    }

    // ---------------- objects tab ----------------
    private async void LoadRenders()
    {
        RenderList.Children.Clear();
        try
        {
            var body = await _api.GetJson("/api/scenes");
            var jobs = await _api.Jobs();
            var available = new HashSet<int>();
            var urlByJob = new Dictionary<int, string>();
            foreach (var sc in body.EnumerateArray())
            {
                var name = sc.GetProperty("name").GetString();
                if (!name.StartsWith("obj_job_")) continue;
                if (int.TryParse(name.Substring(8), out var v))
                {
                    available.Add(v);
                    urlByJob[v] = sc.GetProperty("url").GetString();
                }
            }
            // one entry per OBJECT (that still exists): its newest finished reconstruction
            var live = new HashSet<int>(_objects.Select(o => o.id));
            var latest = jobs.Where(j => j.status == "done" && j.kind != "map" && available.Contains(j.id)
                                         && live.Contains(j.object_id))
                             .GroupBy(j => j.object_id)
                             .Select(g => g.OrderByDescending(j => j.id).First())
                             .OrderByDescending(j => j.id);
            foreach (var j in latest)
            {
                var o = _objects.FirstOrDefault(x => x.id == j.object_id);
                var title = o != null ? o.label : $"object {j.object_id}";
                var jid = j.id;
                var url = urlByJob[jid];
                var btn = new Button
                {
                    Content = title,
                    HorizontalContentAlignment = HorizontalAlignment.Left,
                    HorizontalAlignment = HorizontalAlignment.Stretch,
                    Margin = new Thickness(2),
                    ToolTip = $"job #{jid}",
                };
                System.Windows.Automation.AutomationProperties.SetName(btn, $"OBJ_{jid}");
                async Task Open()
                {
                    try
                    {
                        await Web2.EnsureCoreWebView2Async();
                        // gaussian close-up; the page itself falls back to the point view if no splat pack.
                        // A fresh query string forces a real reload (same URL = WebView2 keeps the old page
                        // and its exposure/orbit state) and defeats the HTTP cache after a repack.
                        Web2.Source = new Uri(_api.Base + url.Replace("index.html", "splat.html") +
                                              "?v=" + DateTime.UtcNow.Ticks);
                        LblRender.Text = $"{title} · job #{jid}";
                    }
                    catch (Exception ex) { LblRender.Text = ex.Message; }
                }
                btn.Click += async (_, _) => await Open();
                RenderList.Children.Add(btn);
                if (RenderList.Children.Count == 1 && Web2.Source == null) await Open();   // auto-open newest
            }
        }
        catch { }
        if (RenderList.Children.Count == 0)
            LblRender.Text = "No generated objects yet — run a reconstruction from the recorder";
    }

    private void ReloadRenders(object s, RoutedEventArgs e) => LoadRenders();
}
