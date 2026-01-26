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
    /// ImageButton.xaml 的交互逻辑
    /// </summary>
    public partial class ImageButton : UserControl
    {
        public ImageButton()
        {
            InitializeComponent();
        }
        public event RoutedEventHandler Click;
        private void UserControl_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
        }

        public void SetUI(string path)
        {
            Uri imageUri = new Uri(AppDomain.CurrentDomain.BaseDirectory + path);
            BitmapImage bitmapImage = new BitmapImage(imageUri);
            if (img != null)
            {
                img.Source = bitmapImage;
            }
        }

    }
}
