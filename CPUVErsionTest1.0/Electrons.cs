using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CPUVErsionTest1._0
{
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
