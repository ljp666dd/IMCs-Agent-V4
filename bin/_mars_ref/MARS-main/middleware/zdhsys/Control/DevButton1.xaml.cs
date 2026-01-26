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
    /// DevButton.xaml 的交互逻辑
    /// </summary>
    public partial class DevButton1 : UserControl
    {
        public DevButton1()
        {
            InitializeComponent();
        }

        public event RoutedEventHandler Click;

        /// <summary>
        /// 设置按钮样式
        /// </summary>
        /// <param name="hexBG">按钮背景色</param>
        /// <param name="hexBd">按钮边框色</param>
        /// <param name="path">图标相对路径</param>
        /// <param name="hexTxt">按钮文本颜色</param>
        /// <param name="txt">按钮文本</param>
        public void SetUI(string hexBG,string hexBd,string path,string hexTxt,string txt)
        {
            Color color = (Color)ColorConverter.ConvertFromString(hexBG);
            SolidColorBrush brush = new SolidColorBrush(color);
            bd.Background = brush;

            Color color1 = (Color)ColorConverter.ConvertFromString(hexBd);
            SolidColorBrush brush1 = new SolidColorBrush(color1);
            bd.BorderBrush = brush1;

            Uri imageUri = new Uri(AppDomain.CurrentDomain.BaseDirectory + path);
            BitmapImage bitmapImage = new BitmapImage(imageUri);
            if (img != null)
            {
                img.Source = bitmapImage;
            }

            Color color2 = (Color)ColorConverter.ConvertFromString(hexTxt);
            SolidColorBrush brush2 = new SolidColorBrush(color2);
            if (tbk != null)
            {
                tbk.Foreground = brush2;
                tbk.Content = txt;
            }
        }

        private void UserControl_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
        }
    }
}
