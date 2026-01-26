using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace zdhsys.Control
{
    /// <summary>
    /// DevOperation.xaml 的交互逻辑
    /// </summary>
    public partial class DevOperation3 : UserControl
    {
        public DevOperation3()
        {
            InitializeComponent();
        }
        public event RoutedEventHandler Click_Up;
        public event RoutedEventHandler Click_Down;
        public object obj;
        private readonly string move_up = "上移";
        private readonly string move_down = "下移";
        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            btn_del.SetUI("\\Image\\move_up.png", "#FF0000", move_up);
            btn_edit.SetUI("\\Image\\move_down.png", "#027AFF", move_down);

            btn_del.Click += Btn_Click;
            btn_edit.Click += Btn_Click;
        }

        private void Btn_Click(object sender, RoutedEventArgs e)
        {
            DevButton2 btn = sender as DevButton2;
            if (btn.Text == move_up)
            {
                if (Click_Up != null)
                {
                    Click_Up.Invoke(this, e);
                }
            }
            else if (btn.Text == move_down)
            {
                if (Click_Down != null)
                {
                    Click_Down.Invoke(this, e);
                }
            }
        }
    }
}
