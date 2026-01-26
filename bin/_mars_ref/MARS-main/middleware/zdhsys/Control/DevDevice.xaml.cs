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
using zdhsys.Popup;
using zdhsys.Unitils;

namespace zdhsys.Control
{
    /// <summary>
    /// DevDevice.xaml 的交互逻辑
    /// </summary>
    public partial class DevDevice : UserControl
    {
        public DevDevice()
        {
            InitializeComponent();
        }

        public string DevName = "";

        public void SetUI(string devName,string path)
        {
            Uri imageUri = new Uri(AppDomain.CurrentDomain.BaseDirectory + path);
            BitmapImage bitmapImage = new BitmapImage(imageUri);
            img.Source = bitmapImage;
            dev_name.Content = devName;
            DevName = devName;
        }

        HomeDeviceInfo info;

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

        private void UserControl_MouseEnter(object sender, MouseEventArgs e)
        {
            if (info != null)
            {
                info.Close();
                info = null;
            }
            Point p = GlobalUnitils.GetScreenPosition(this);
            //Console.WriteLine(p.X + " - " + p.Y);
            info = new HomeDeviceInfo(p);
            info.SetUI(DevName, "断开", "POS", "无");
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
