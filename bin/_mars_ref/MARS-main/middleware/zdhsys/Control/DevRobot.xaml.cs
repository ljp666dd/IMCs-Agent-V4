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
using zdhsys.entity;
using zdhsys.Popup;
using zdhsys.Unitils;

namespace zdhsys.Control
{
    /// <summary>
    /// DevRobot.xaml 的交互逻辑
    /// </summary>
    public partial class DevRobot : UserControl
    {
        public DevRobot()
        {
            InitializeComponent();
        }

        public void SetUI(string name, string status, string path)
        {
            Uri imageUri = new Uri(AppDomain.CurrentDomain.BaseDirectory + path);
            BitmapImage bitmapImage = new BitmapImage(imageUri);
            img.Source = bitmapImage;
            lbName.Content = name;
            lbStatus.Content = status;
            DevName = name;
        }
        HomeDeviceInfo info;
        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {

        }

        private void UserControl_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (info != null)
            {
                info.Close();
                info = null;
            }
            else
            {
                Point p = GlobalUnitils.GetScreenPosition(this);
                Console.WriteLine(p.X + " - " + p.Y);
                info = new HomeDeviceInfo(p);
                info.Show();
            }
        }

        public string DevName = "";
        public Heart ht;
        public RobotHeart rh;

        private void UserControl_MouseEnter(object sender, MouseEventArgs e)
        {
            if (info != null)
            {
                info.Close();
                info = null;
            }
            Point p = GlobalUnitils.GetScreenPosition(this);
            info = new HomeDeviceInfo(p);
            if (ht != null)
            {
                info.SetUI(DevName, ht.rob_sta == 0 ? "空闲" : "执行中", ht.ID + "", rh.cmd + "");
            }
            else
            {
                info.SetUI(DevName, "断开", "", "");
            }
            info.Show();
        }

        private void UserControl_MouseLeave(object sender, MouseEventArgs e)
        {
            if (info != null)
            {
                info.Close();
                info = null;
            }
        }
    }
}
