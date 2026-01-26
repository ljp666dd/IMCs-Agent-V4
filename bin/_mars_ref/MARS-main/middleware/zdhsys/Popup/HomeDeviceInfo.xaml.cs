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
using System.Windows.Shapes;

namespace zdhsys.Popup
{
    /// <summary>
    /// HomeDeviceInfo.xaml 的交互逻辑
    /// </summary>
    public partial class HomeDeviceInfo : Window
    {
        public HomeDeviceInfo(Point p)
        {
            InitializeComponent();
            Left = p.X;
            Top = p.Y - Height;
        }

        public void SetUI(string name,string status,string location,string cmd)
        {
            lbCmd.Content = cmd;
            lbLocation.Content = location;
            lbName.Content = name;
            lbStatus.Content = status;
        }

        private void Window_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            Close();
        }
    }
}
