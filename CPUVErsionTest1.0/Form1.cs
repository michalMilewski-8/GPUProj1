#define BY_GPU

using Alea;
using Alea.CSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;


namespace CPUVErsionTest1._0
{
    class Elektrons
    {
        public List<int> elektrons_x;
        public List<int> elektrons_y;
        public List<int> elektrons_charge;
        public List<int> elektrons_move_x;
        public List<int> elektrons_move_y;

        public Elektrons()
        {
            elektrons_x = new List<int>();
            elektrons_y = new List<int>();
            elektrons_charge = new List<int>();
            elektrons_move_x = new List<int>();
            elektrons_move_y = new List<int>();
        }

        public void Add(int x, int y, int charge, int move_x,int move_y)
        {
            elektrons_x.Add(x);
            elektrons_y.Add(y);
            elektrons_charge.Add(charge);
            elektrons_move_x.Add(move_x);
            elektrons_move_y.Add(move_y);
        }

        public void ToArray(out int[] x, out int[] y, out int[] charge, out int [] move_x, out int [] move_y) 
        {
            x = elektrons_x.ToArray();
            y = elektrons_y.ToArray();
            charge = elektrons_charge.ToArray();
            move_x = elektrons_move_x.ToArray();
            move_y = elektrons_move_y.ToArray();
        }

        public void FromArray(int[] x, int[] y, int[] charge, int [] move_x, int [] move_y)
        {
            elektrons_x = new List<int>(x);
            elektrons_y = new List<int>(y);
            elektrons_charge = new List<int>(charge);
            elektrons_move_x = new List<int>(move_x);
            elektrons_move_y = new List<int>(move_y);
        }
    }
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            elektrons = new Elektrons();
            Random rand = new Random();
            for(int i = 0; i < elektron_count; i++)
            {

                int x = rand.Next(0, drawing_panel.Width);
                int y = rand.Next(0, drawing_panel.Height);
                int charge = rand.Next(-max_charge,max_charge);
                int move_x = rand.Next(0,15);
                int move_y = rand.Next(0,15);

                elektrons.Add(x, y, charge, move_x, move_y);
            }
            values = new float[drawing_panel.Width, drawing_panel.Height];
            solve_by_GPU = true;
        }

        private Elektrons elektrons;
        private int elektron_count = 100;
        private int max_charge = 100;
        private float[,] values;
        private readonly int Bias = 5;
        private int moving_elektron_indeks = -1;
        private bool solve_by_GPU = false;

        private float ComputeCharge(int x, int y, Elektrons elektrons_)
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

        unsafe private void ShowResult(Graphics e)
        {
            Stopwatch sw = new Stopwatch();
            Stopwatch sw1 = new Stopwatch();
            sw.Start();
            var pic = new Bitmap(drawing_panel.Width, drawing_panel.Height);
           
            float max = 10;
            if (!solve_by_GPU)
            {
                var modified_pic = new BmpPixelSnoop(pic);
                for (int i = 0; i < drawing_panel.Width; i++)
                {
                    for (int j = 0; j < drawing_panel.Height; j++)
                    {
                        values[i, j] = ComputeCharge(i, j, elektrons);
                        var col = MapRainbowColor((values[i, j] + 1), max, -max);
                        modified_pic.SetPixel(i, j, col.r, col.g, col.b);
                    }
                }
                modified_pic.Dispose();
            }
            else
            {
                byte[] mod;
                
                SolveByGpu(elektrons, drawing_panel.Width, drawing_panel.Height, out mod);
                

                fixed (byte* ptr = mod)
                {
                    pic = new Bitmap(drawing_panel.Width, drawing_panel.Height, 4*drawing_panel.Width,
                                    PixelFormat.Format32bppArgb, new IntPtr(ptr));
                }
                
                
            }

            //drawing_panel.Image = pic;
            //pic.Dispose();
            sw1.Start();
            e.DrawImageUnscaled(pic, 0, 0);
            //drawing_panel.Image = pic;

            label1.Text = sw1.ElapsedMilliseconds.ToString();
            foreach (var el in elektrons)
            {
                e.FillEllipse(Brushes.LightGray, el.x - Bias, el.y - Bias, 2 * Bias, 2 * Bias);
                if (el.charge > 0)
                    e.DrawString("+", SystemFonts.DefaultFont, Brushes.Black, el.x - Bias, el.y - Bias - 2);
                else
                    e.DrawString("-", SystemFonts.DefaultFont, Brushes.Black, el.x - Bias, el.y - Bias - 2);
            }
            sw1.Stop();
            sw.Stop();
            Text = sw.ElapsedMilliseconds.ToString() + " ms";
        }

        private static void Kernel(byte[] result_r, int[] elektron_x, int[] elektron_y, int[] elektron_charge, int[] elektron_move_x, int[] elektron_move_y, int width,int height)
        {
            var start_s = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (int start = start_s; start < result_r.Length/4; start += stride)
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
                float min = -30;
                float max = 30;

                float f;
                if (value < min) value = min;
                if (value > max) value = max;
                f = value - min;
                f /= (max - min);

                float a = (1 - f) / 0.2f;
                var X = (int)a;
                var Y = (byte)(255 * (a - X));
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

                result_r[4*start] = b;
                result_r[4*start+1] = g;
                result_r[4*start+2] = r;
                result_r[4*start+3] = 255;
            }
        }
        [GpuManaged]
        private static void SolveByGpu(Elektrons elektrons_, int width, int height, out byte[] snoop)
        {
            var gpu = Gpu.Default;
            var lp = new LaunchParam(128, 1024);

            int[] elektron_x;
            int[] elektron_y;
            int[] elektron_charge;
            int[] elektron_move_x;
            int[] elektron_move_y;

            elektrons_.ToArray(out elektron_x, out elektron_y, out elektron_charge, out elektron_move_x, out elektron_move_y);


            var result_r = new byte[4* width * height];

            int dwidth = width;
            int dheight = height;

            gpu.Launch(Kernel, lp, result_r, elektron_x, elektron_y, elektron_charge, elektron_move_x, elektron_move_y, dwidth, dheight);

            //Parallel.For(0, result_r.Length - 1, (int i) => //for (int i = 0; i < result_r.Length; i++)
            //{
            //    int x = i % width;
            //    int y = i / width;
            //    snoop.SetPixel(x, y, result_r[i], result_g[i], result_b[i]);
            //});

            elektrons_.FromArray(elektron_x, elektron_y, elektron_charge, elektron_move_x, elektron_move_y);

            snoop = result_r;

        }

        private int FindIfOnVertex(Point p)
        {
            return elektrons.FindIndex((Elektron e) => { return Math.Abs(e.x - p.X) < Bias && Math.Abs(e.y - p.Y) < Bias; });
        }

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

        #region callbacks
        private void drawing_panel_Paint(object sender, PaintEventArgs e)
        {
            
            ShowResult(e.Graphics);
            
        }

        private void drawing_panel_SizeChanged(object sender, EventArgs e)
        {
            values = new float[drawing_panel.Width, drawing_panel.Height];
        }

        private void drawing_panel_MouseDown(object sender, MouseEventArgs e)
        {
            //moving_elektron_indeks = FindIfOnVertex(e.Location);
        }

        private void drawing_panel_MouseMove(object sender, MouseEventArgs e)
        {
            //if (moving_elektron_indeks >= 0)
            //{
            //    elektrons[moving_elektron_indeks] = new Elektron(e.X, e.Y, elektrons[moving_elektron_indeks].charge);
            //    drawing_panel.Invalidate();
            //}
        }

        private void drawing_panel_MouseUp(object sender, MouseEventArgs e)
        {
            //moving_elektron_indeks = -1;
        }

        private void use_gpu_CheckedChanged(object sender, EventArgs e)
        {
            solve_by_GPU = use_gpu.Checked;
            drawing_panel.Invalidate();

        }
        #endregion
    }
}
