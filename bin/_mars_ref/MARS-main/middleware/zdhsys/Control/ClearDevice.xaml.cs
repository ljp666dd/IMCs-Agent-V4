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
    /// ClearDevice.xaml 的交互逻辑
    /// </summary>
    public partial class ClearDevice : UserControl
    {
        public ClearDevice()
        {
            InitializeComponent();
        }
        public void SetUI(string name,string status,string path)
        {
            Uri imageUri = new Uri(AppDomain.CurrentDomain.BaseDirectory + path);
            BitmapImage bitmapImage = new BitmapImage(imageUri);
            img.Source = bitmapImage;
            lbName.Content = name;
            lbStatus.Content = status;
        }
    }
}
