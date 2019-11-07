#define BY_GPU

using Alea;
using Alea.CSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;


namespace CPUVErsionTest1._0
{

    struct Elektron
    {
        public int x;
        public int y;
        public int charge;
        public Elektron(int x_, int y_, int charge_)
        {
            x = x_;
            y = y_;
            charge = charge_;
        }
    }
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            elektrons = new List<Elektron>();
            elektrons.Add(new Elektron(200, 200, 29700));
            elektrons.Add(new Elektron(200, 300, -3010));
            elektrons.Add(new Elektron(300, 210, 3000));
            elektrons.Add(new Elektron(220, 220, 3000));
            elektrons.Add(new Elektron(220, 320, 3000));
            for(int i = 0; i < 0;i++)
                elektrons.Add(new Elektron(320, 520, 3000));

            values = new float[drawing_panel.Width, drawing_panel.Height];
            solve_by_GPU = true;
            drawing_panel.Invalidate();
        }

        private List<Elektron> elektrons;
        private float[,] values;
        private readonly int Bias = 5;
        private int moving_elektron_indeks = -1;
        private bool solve_by_GPU = false;

        private float ComputeCharge(int x, int y, List<Elektron> elektrons_)
        {
            float result = 0;
            foreach (var e in elektrons_)
            {
                result += (float)(e.charge / (float)Math.Pow(Len(x, y, e.x, e.y), 2));
                if (x == e.x && y == e.y) return 0;
            }
            return result;
        }

        private float Len(int x, int y, int cx, int cy)
        {
            int diffx = x - cx;
            int diffy = y - cy;

            return (float)Math.Sqrt(diffx * diffx + diffy * diffy);
        }

        private void ShowResult(Graphics e)
        {
            var pic = new Bitmap(drawing_panel.Width, drawing_panel.Height);
            var modified_pic = new BmpPixelSnoop(pic);
            float max = 10;
            if (!solve_by_GPU)
            {
               for (int i = 0; i < drawing_panel.Width; i++)
                {
                    for (int j = 0; j < drawing_panel.Height; j++)
                    {
                        values[i, j] = ComputeCharge(i, j, elektrons);
                        var col = MapRainbowColor((values[i, j] + 1), max, -max);
                        modified_pic.SetPixel(i, j, col.r, col.g, col.b);
                    }
                }
            }
            else
                SolveByGpu(elektrons, modified_pic.Width, modified_pic.Height, modified_pic);
            modified_pic.Dispose();
            //drawing_panel.Image = pic;
            //pic.Dispose();
            e.DrawImage(pic, 0, 0);
            foreach(var el in elektrons)
            {
                e.FillEllipse(Brushes.LightGray, el.x - Bias, el.y - Bias, 2 * Bias, 2 * Bias);
                if (el.charge > 0)
                    e.DrawString("+", SystemFonts.DefaultFont, Brushes.Black, el.x - Bias, el.y - Bias-2);
                else
                    e.DrawString("-", SystemFonts.DefaultFont, Brushes.Black, el.x - Bias, el.y - Bias-2);
            }
        }

#if (BY_GPU)
        private static void Kernel(byte[] result_r, byte[] result_g, byte[] result_b, int[] elektron_x, int[] elektron_y, int[] elektron_charge,int width)
#else
        private static void Kernel(float[] result_r, int[] result_g, byte[] result_b, int[] elektron_x, int[] elektron_y, int[] elektron_charge, int width)
#endif
        {
            var start_s = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (int start = start_s; start < result_r.Length; start += stride)
            {
                int x = start % width;
                int y = start / width;
                float result = 0;
                for (int i = 0; i < elektron_charge.Length; i++)
                {
                    if (x == elektron_x[i] && y == elektron_y[i])
                    {
                        result = 0;
                        break;
                    }
                    int diffx = x - elektron_x[i];
                    int diffy = y - elektron_y[i];
                    float len = (float)Math.Sqrt(diffx * diffx + diffy * diffy);
                    result += (float)elektron_charge[i] / (len * len);
                }
                float value = result;
                float min = -10;
                float max = 10;

                float f;
                if (value < min) value = min;
                if (value > max) value = max;
                f = value - min;
                f /= (max - min);

                float a = (1 - f) / 0.2f;
                var X = (int)a;
                var Y = (byte)(255 * (a - X));
                //switch (X)
                //{
                //    case 0: r = 255; g = Y; b = 0; break;
                //    case 1: r = 255 - Y; g = 255; b = 0; break;
                //    case 2: r = 0; g = 255; b = Y; break;
                //    case 3: r = 0; g = 255 - Y; b = 255; break;
                //    case 4: r = Y; g = 0; b = 255; break;
                //    case 5: r = 255; g = 0; b = 255; break;
                //}
#if (BY_GPU)
                byte r = 0, g = 0, b = 0;
                if (X == 0)
                {
                    r = 255; g = Y; b = 0;
                }
                else if (X == 1)
                {
                    r = (byte)(255 - Y); g = 255; b = 0;
                }
                else if (X == 2)
                {
                    r = 0; g = 255; b = Y;
                }
                else if (X == 3)
                {
                    r = 0; g = (byte)(255 - Y); b = 255;
                }
                else if (X == 4)
                {
                    r = Y; g = 0; b = 255;
                }
                else
                {
                    r = 255; g = 0; b = 255;
                }

                result_r[start] = r;
                result_g[start] = g;
                result_b[start] = b;
#else
                result_r[start] = a;
                result_g[start] = X;
                result_b[start] = Y;
#endif
            }
        }
        [GpuManaged]
        private static void SolveByGpu(List<Elektron> elektrons_, int width, int height, BmpPixelSnoop snoop)
        {
            var gpu = Gpu.Default;
            var lp = new LaunchParam(512, 1024);
            var elektron_x = elektrons_.ConvertAll<int>((Elektron e) => e.x).ToArray();
            var elektron_y = elektrons_.ConvertAll<int>((Elektron e) => e.y).ToArray();
            var elektron_charge = elektrons_.ConvertAll<int>((Elektron e) => e.charge).ToArray();

#if (BY_GPU)
            var result_r = new byte[width * height];
            var result_g = new byte[width * height];
            var result_b = new byte[width * height];
#else
            var result_r = new float[width * height];
            var result_g = new int[width * height];
            var result_b = new byte[width * height];
#endif
            int dwidth = width;

            gpu.Launch(Kernel, lp, result_r, result_g, result_b, elektron_x, elektron_y, elektron_charge, dwidth);

            Parallel.For(0, result_r.Length - 1, (int i) => //for (int i = 0; i < result_r.Length; i++)
            {
#if (BY_GPU)
                int x = i % width;
                int y = i / width;
                snoop.SetPixel(x, y, result_r[i], result_g[i], result_b[i]);
#else

                int x;
                int y;
                byte r = 0, g = 0, b = 0;
                int X = 0;
                byte Y = 0;
                float a = 0;
                x = i % width;
                y = i / width;
                X = result_g[i];
                Y = result_b[i];
                a = result_r[i];

                if (X == 0)
                {
                    r = 255; g = Y; b = 0;
                }
                else if (X == 1)
                {
                    r = (byte)(255 - Y); g = 255; b = 0;
                }
                else if (X == 2)
                {
                    r = 0; g = 255; b = Y;
                }
                else if (X == 3)
                {
                    r = 0; g = (byte)(255 - Y); b = 255;
                }
                else if (X == 4)
                {
                    r = Y; g = 0; b = 255;
                }
                else
                {
                    r = 255; g = 0; b = 255;
                }


                snoop.SetPixel(x, y, r, g, b);
#endif
            });
        }

        private int FindIfOnVertex(Point p)
        {
            return elektrons.FindIndex((Elektron e) => { return Math.Abs(e.x - p.X) < Bias && Math.Abs(e.y - p.Y) < Bias; });
        }

        // Map a value to a rainbow color.
        private (byte r, byte g, byte b) MapRainbowColor(float value, float max, float min)
        {
            byte r = 0, g = 0, b = 0;
            float f;
            if (value < min) value = min;
            if (value > max) value = max;
            f = value - min;
            f = f / (max - min);

            float a = (1 - f) / 0.2f;
            var X = (byte)a;
            var Y = (byte)(255 * (a - X));
            switch (X)
            {
                case 0: r = 255; g = Y; b = 0; break;
                case 1: r = (byte)(255 - Y); g = 255; b = 0; break;
                case 2: r = 0; g = 255; b = Y; break;
                case 3: r = 0; g = (byte)(255 - Y); b = 255; break;
                case 4: r = Y; g = 0; b = 255; break;
                case 5: r = 255; g = 0; b = 255; break;
            }

            return (r, g, b);
        }

        private void drawing_panel_Paint(object sender, PaintEventArgs e)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            ShowResult(e.Graphics);
            sw.Stop();
            Text = sw.ElapsedMilliseconds.ToString() + " ms";
        }

        private void drawing_panel_SizeChanged(object sender, EventArgs e)
        {
            values = new float[drawing_panel.Width, drawing_panel.Height];
        }

        private void drawing_panel_MouseDown(object sender, MouseEventArgs e)
        {
            moving_elektron_indeks = FindIfOnVertex(e.Location);
        }

        private void drawing_panel_MouseMove(object sender, MouseEventArgs e)
        {
            if (moving_elektron_indeks >= 0)
            {
                elektrons[moving_elektron_indeks] = new Elektron(e.X, e.Y, elektrons[moving_elektron_indeks].charge);
                drawing_panel.Invalidate();
            }
        }

        private void drawing_panel_MouseUp(object sender, MouseEventArgs e)
        {
            moving_elektron_indeks = -1;
        }

        private void use_gpu_CheckedChanged(object sender, EventArgs e)
        {
            solve_by_GPU = use_gpu.Checked;
            drawing_panel.Invalidate();

        }
    }
}
