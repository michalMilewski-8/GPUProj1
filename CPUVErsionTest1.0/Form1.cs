using Alea;
using Alea.CSharp;
using Alea.Parallel;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;


namespace CPUVErsionTest1._0
{

    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            GenerateElectrons();
            solve_by_GPU = true;
            ShowResult();
        }

        private Electrons electrons;
        private int electron_count = 1000;
        private bool solve_by_GPU = false;
        private byte[] r_val;
        private byte[] g_val;
        private byte[] b_val;


        private void GenerateElectrons()
        {
            electrons = new Electrons();
            Random rand = new Random();
            for (int i = 0; i < electron_count; i++)
            {

                int x = rand.Next(0, drawing_panel.Width);
                int y = rand.Next(0, drawing_panel.Height);
                int move_x = rand.Next(-10, 10);
                int move_y = rand.Next(-10, 10);
                short charge = rand.Next(0, 10) % 2 == 0 ? (short)-1 : (short)1;

                electrons.Add(x, y, move_x, move_y, charge);
            }
            CreateColorValues();
            electrons.MakeArrays();
        }
        private float ComputeCharge(int x, int y, Electrons electrons_)
        {
            float result = 0;
            for (int i = 0; i < electrons_.electrons_x_.Length; i++)
            {
                result += (float)(1 / (float)Len(x, y, electrons_.electrons_x_[i], electrons_.electrons_y_[i]))*electrons_.electrons_charge_[i];
                if (x == electrons_.electrons_x_[i] && y == electrons_.electrons_y_[i]) return 0;
            }
            return result;


        }

        private float Len(int x, int y, int cx, int cy)
        {
            int diffx = x - cx;
            int diffy = y - cy;

            return (float)(diffx * diffx + diffy * diffy);
        }

        private void MoveElectrons()
        {
            for (int i = 0; i < electrons.electrons_x_.Length; i++)
            {

                int el_x = electrons.electrons_x_[i];
                int el_y = electrons.electrons_y_[i];
                el_x += electrons.electrons_move_x_[i];
                el_y += electrons.electrons_move_y_[i];

                if (el_y >= drawing_panel.Height || el_y < 0)
                {
                    electrons.electrons_move_y_[i] *= -1;
                }
                else
                {
                    electrons.electrons_y_[i] = el_y;
                }
                if (el_x >= drawing_panel.Width || el_x < 0)
                {
                    electrons.electrons_move_x_[i] *= -1;
                }
                else
                {

                    electrons.electrons_x_[i] = el_x;
                }

            }
        }

        unsafe private void ShowResult()
        {
            Stopwatch sw = new Stopwatch();
            Stopwatch sw1 = new Stopwatch();
            sw.Start();
            Bitmap pic;
            float max = 0.1f;
            sw1.Start();
            if (!solve_by_GPU)
            {
                
                pic = new Bitmap(drawing_panel.Width, drawing_panel.Height);
                MoveElectrons();
                var modified_pic = new BmpPixelSnoop(pic);
                Parallel.For(0, drawing_panel.Width, i =>
                {
                    for (int j = 0; j < drawing_panel.Height; j++)
                    {
                        var col = MapRainbowColor(ComputeCharge(i, j, electrons), max, -max);
                        modified_pic.SetPixel(i, j, col.r, col.g, col.b);
                    }
                });
                modified_pic.Dispose();
            }
            else
            {
                byte[] mod;
                string ts;
                SolveByGpu(electrons, drawing_panel.Width, drawing_panel.Height, out mod, out ts, r_val, g_val, b_val);

                
                fixed (byte* ptr = mod)
                {
                    pic = new Bitmap(drawing_panel.Width, drawing_panel.Height, 4 * drawing_panel.Width,
                                    PixelFormat.Format32bppArgb, new IntPtr(ptr));
                }
                

            }
            sw1.Stop();


            drawing_panel.Image = pic;

            
            label1.Text = sw1.ElapsedMilliseconds.ToString() + " ms";

            sw.Stop();
            Text = sw.ElapsedMilliseconds.ToString() + " ms";
        }
        private static void Kernel_move_electrons(int[] electron_x, int[] electron_y, int[] electron_move_x, int[] electron_move_y, int width, int height)
        {
            var start_s = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;

            for (int i = start_s; i < electron_x.Length; i += stride)
            {

                int el_x = electron_x[i];
                int el_y = electron_y[i];
                el_x += electron_move_x[i];
                el_y += electron_move_y[i];

                if (el_y >= height || el_y < 0)
                {
                    electron_move_y[i] *= -1;
                }
                else
                {
                    electron_y[i] = el_y;
                }
                if (el_x >= width || el_x < 0)
                {
                    electron_move_x[i] *= -1;
                }
                else
                {
                    electron_x[i] = el_x;
                }
            }
        }


        private static void Kernel(byte[] result_r, int[] electron_x, int[] electron_y, short[] charge, int width, byte[] r_val, byte[] g_val, byte[] b_val)
        {
            var kolorki = Intrinsic.__address_of_array(__shared__.ExternArray<byte>());
            var electrons = (kolorki + r_val.Length*3).Reinterpret<byte>();

            int start_s = threadIdx.x + threadIdx.y * blockDim.x;
            int stride = blockDim.x * blockDim.y;
            int block_x = blockIdx.x * blockDim.x;
            int block_y = blockIdx.y * blockDim.y;

            for (int i = start_s; i < electron_x.Length; i += stride)
            {
                int diffx = block_x - electron_x[i];
                int diffy = block_y - electron_y[i];
                if (diffx > 100 || diffx < -100 || diffy > 100 || diffy < -100)
                {
                    electrons[i] = 0;
                }
                else
                {
                    electrons[i] = 1;
                }
            }

            for (int i = start_s; i < r_val.Length; i += stride)
            {
                kolorki[3 * i] = r_val[i];
                kolorki[3 * i + 1] = g_val[i];
                kolorki[3 * i + 2] = b_val[i];
            }


            int x = block_x + threadIdx.x;
            int y = block_y + threadIdx.y;

            float result = 0;
            for (int i = 0; i < electron_x.Length; i++)
            {
                if (electrons[i] == 1)
                {
                    if (x == electron_x[i] && y == electron_y[i])
                    {
                        result = 0;
                        break;
                    }
                    int diffx = x - electron_x[i];
                    int diffy = y - electron_y[i];

                    if (diffx < 100 && diffx > -100 && diffy < 100 && diffy > -100)
                    {
                        float len = (float)(diffx * diffx + diffy * diffy);
                        result += 1 / (len * charge[i]);
                    }
                }
            }

            float value = result;
            float min = -0.1f;
            float max = 0.1f;

            float f;
            if (value < min) value = min;
            if (value > max) value = max;
            f = value - min;
            f /= (max - min);


            
            int start = y * width + x;
            int col = ((int)(f * 1023));
            result_r[4 * start] = b_val[col];
            result_r[4 * start + 1] = g_val[col];
            result_r[4 * start + 2] = r_val[col];
            result_r[4 * start + 3] = 255;

            //int col = ((int)(f * 1023)) * 3 + 2;
            //result_r[4 * start] = kolorki[col--];
            //result_r[4 * start + 1] = kolorki[col--];
            //result_r[4 * start + 2] = kolorki[col];
            //result_r[4 * start + 3] = 255;

        }

        [GpuManaged]
        private static void SolveByGpu(Electrons electrons_, int width, int height, out byte[] snoop, out string str, byte[] r, byte[] g, byte[] b)
        {
            Stopwatch sw = new Stopwatch();
            var gpu = Gpu.Default;
            var block_dim = new dim3(32, 32);
            var grid_dim = new dim3(width / 32,  height / 32);

            var lp = new LaunchParam(grid_dim, block_dim, r.Length * 3 + electrons_.electrons_y_.Length);
            var lp_move = new LaunchParam(electrons_.electrons_y_.Length % 1024, 1024);

            int[] electron_x;
            int[] electron_y;
            int[] electron_move_x;
            int[] electron_move_y;
            short[] charge;

            electrons_.ToArray(out electron_x, out electron_y, out electron_move_x, out electron_move_y, out charge);

            var result_r = new byte[4 * width * height];

            int dwidth = width;
            int dheight = height;

            ///

            gpu.Launch(Kernel, lp, result_r, electron_x, electron_y, charge, dwidth, r, g, b);


            gpu.Launch(Kernel_move_electrons, lp_move, electron_x, electron_y, electron_move_x, electron_move_y, dwidth, dheight);

            ///


            electrons_.FromArray(electron_x, electron_y, electron_move_x, electron_move_y, charge);

            snoop = result_r;


            str = sw.ElapsedMilliseconds.ToString() + " ms";
        }

        private (byte r, byte g, byte b) MapRainbowColor(float value, float max, float min)
        {
            byte r = 0, g = 0, b = 0;
            float f;
            if (value < min) value = min;
            if (value > max) value = max;
            f = value - min;
            f /= (max - min);

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

        private void CreateColorValues()
        {
            r_val = new byte[1024];
            g_val = new byte[1024];
            b_val = new byte[1024];
            for (int i = 0; i < 1024; i++)
            {
                var res = MapRainbowColor(i, 1023, 0);
                r_val[i] = res.r;
                g_val[i] = res.g;
                b_val[i] = res.b;
            }
        }

        #region callbacks

        private void drawing_panel_SizeChanged(object sender, EventArgs e)
        {
            if (drawing_panel.Width != 0 && drawing_panel.Height != 0)
            {
                GenerateElectrons();
                ShowResult();
            }
        }

        private void use_gpu_CheckedChanged(object sender, EventArgs e)
        {
            solve_by_GPU = use_gpu.Checked;
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
                ShowResult();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (!timer1.Enabled)
                timer1.Start();
            else
                timer1.Stop();
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            timer1.Stop();
            int old = electron_count;
            if (int.TryParse(textBox1.Text, out electron_count) && electron_count > 0)
                GenerateElectrons();
            else
                electron_count = old;

        }

        #endregion
    }

    class Electrons
    {
        public List<int> electrons_x;
        public List<int> electrons_y;
        public List<int> electrons_move_x;
        public List<int> electrons_move_y;
        public List<short> electrons_charge;
        public int[] electrons_x_;
        public int[] electrons_y_;
        public int[] electrons_move_x_;
        public int[] electrons_move_y_;
        public short[] electrons_charge_;

        public Electrons()
        {
            electrons_x = new List<int>();
            electrons_y = new List<int>();
            electrons_move_x = new List<int>();
            electrons_move_y = new List<int>();
            electrons_charge = new List<short>();
        }

        public void Add(int x, int y, int move_x, int move_y, short charge)
        {
            electrons_x.Add(x);
            electrons_y.Add(y);
            electrons_move_x.Add(move_x);
            electrons_move_y.Add(move_y);
            electrons_charge.Add(charge);
        }

        public void MakeArrays()
        {
            electrons_x_ = electrons_x.ToArray();
            electrons_y_ = electrons_y.ToArray();
            electrons_move_x_ = electrons_move_x.ToArray();
            electrons_move_y_ = electrons_move_y.ToArray();
            electrons_charge_ = electrons_charge.ToArray();
        }

        public void ToArray(out int[] x, out int[] y, out int[] move_x, out int[] move_y, out short[] charge)
        {
            x = electrons_x_;
            y = electrons_y_;
            move_x = electrons_move_x_;
            move_y = electrons_move_y_;
            charge = electrons_charge_;
        }

        public void FromArray(int[] x, int[] y, int[] move_x, int[] move_y, short[] charge)
        {
            electrons_x_ = x;
            electrons_y_ = y;
            electrons_move_x_ = move_x;
            electrons_move_y_ = move_y;
        }
    }
}
