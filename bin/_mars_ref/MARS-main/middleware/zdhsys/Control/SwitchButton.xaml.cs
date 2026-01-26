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
    /// SwitchButton.xaml 的交互逻辑
    /// </summary>
    public partial class SwitchButton : UserControl
    {
        public SwitchButton()
        {
            InitializeComponent();
        }

        public bool flagLeft = true;
        public event RoutedEventHandler Click;

        private void btn_resp_Click(object sender, RoutedEventArgs e)
        {
            btn_resp.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FFFFFF"));
            bd_resp.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#027AFF"));

            btn_move.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#027AFF"));
            bd_move.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FFFFFF"));

            flagLeft = true;
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
        }

        private void btn_move_Click(object sender, RoutedEventArgs e)
        {
            btn_move.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FFFFFF"));
            bd_move.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#027AFF"));

            btn_resp.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#027AFF"));
            bd_resp.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FFFFFF"));

            flagLeft = false;
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
        }
    }
}
