using System.Windows;

namespace NastDeskview;

// Pose math mirrors inspector/server.py and the web viewer:
// pose = {t, q(wxyz), size:[L,H,W]}, R columns = [forward, up, lateral].
public static class Geo
{
    public static double[,] QuatToR(double[] q)
    {
        double w = q[0], x = q[1], y = q[2], z = q[3];
        return new[,]
        {
            { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w) },
            { 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w) },
            { 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y) },
        };
    }

    public static double[] RToQuat(double[,] r)
    {
        double t = r[0, 0] + r[1, 1] + r[2, 2];
        double w, x, y, z;
        if (t > 0)
        {
            double s = 0.5 / Math.Sqrt(t + 1);
            w = 0.25 / s;
            x = (r[2, 1] - r[1, 2]) * s;
            y = (r[0, 2] - r[2, 0]) * s;
            z = (r[1, 0] - r[0, 1]) * s;
        }
        else
        {
            var d = new[] { r[0, 0], r[1, 1], r[2, 2] };
            int i = Array.IndexOf(d, d.Max());
            int j = (i + 1) % 3, k = (i + 2) % 3;
            double s = 2 * Math.Sqrt(Math.Max(1e-12, 1 + r[i, i] - r[j, j] - r[k, k]));
            var qq = new double[4];
            qq[0] = (r[k, j] - r[j, k]) / s;
            qq[1 + i] = 0.25 * s;
            qq[1 + j] = (r[j, i] + r[i, j]) / s;
            qq[1 + k] = (r[k, i] + r[i, k]) / s;
            (w, x, y, z) = (qq[0], qq[1], qq[2], qq[3]);
        }
        return new[] { w, x, y, z };
    }

    public static double[] Col(double[,] r, int c) => new[] { r[0, c], r[1, c], r[2, c] };

    public static double[][] PoseCorners(Pose ps)
    {
        var r = QuatToR(ps.q);
        double L = ps.size[0] / 2, H = ps.size[1] / 2, W = ps.size[2] / 2;
        var outp = new List<double[]>();
        foreach (var a in new[] { -1.0, 1.0 })
            foreach (var b in new[] { -1.0, 1.0 })
                foreach (var d in new[] { -1.0, 1.0 })
                    outp.Add(new[]
                    {
                        ps.t[0] + r[0, 0] * a * L + r[0, 1] * b * H + r[0, 2] * d * W,
                        ps.t[1] + r[1, 0] * a * L + r[1, 1] * b * H + r[1, 2] * d * W,
                        ps.t[2] + r[2, 0] * a * L + r[2, 1] * b * H + r[2, 2] * d * W,
                    });
        return outp.ToArray();
    }

    public static readonly int[][] BoxEdges =
    {
        new[]{0,1}, new[]{0,2}, new[]{1,3}, new[]{2,3},
        new[]{4,5}, new[]{4,6}, new[]{5,7}, new[]{6,7},
        new[]{0,4}, new[]{1,5}, new[]{2,6}, new[]{3,7},
    };

    // World point -> upright-canvas pixel for a given frame pose. Null if behind.
    public static Point? Project(FramePose fr, double[] X, Intrinsics I)
    {
        var r = QuatToR(fr.q);
        double dx = X[0] - fr.p[0], dy = X[1] - fr.p[1], dz = X[2] - fr.p[2];
        double xc = r[0, 0] * dx + r[0, 1] * dy + r[0, 2] * dz;
        double yc = r[1, 0] * dx + r[1, 1] * dy + r[1, 2] * dz;
        double zc = r[2, 0] * dx + r[2, 1] * dy + r[2, 2] * dz;
        if (zc <= 0.05) return null;
        double u = I.fx * xc / zc + I.cx;
        double v = I.fy * yc / zc + I.cy;
        return new Point((I.h - 1) - v, u);        // upright: x=(H-1)-sy, y=sx
    }
}
