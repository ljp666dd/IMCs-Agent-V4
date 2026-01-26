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
    public partial class DevButton2 : UserControl
    {
        public DevButton2()
        {
            InitializeComponent();
        }

        public event RoutedEventHandler Click;

        public string Text;

        /// <summary>
        /// 设置按钮样式
        /// </summary>
        /// <param name="hexBG">按钮背景色</param>
        /// <param name="hexBd">按钮边框色</param>
        /// <param name="path">图标相对路径</param>
        /// <param name="hexTxt">按钮文本颜色</param>
        /// <param name="txt">按钮文本</param>
        public void SetUI(string path,string hexTxt,string txt)
        {
            Text = txt;
            Uri imageUri = new Uri(AppDomain.CurrentDomain.BaseDirectory + path);
            BitmapImage bitmapImage = new BitmapImage(imageUri);
            img.Source = bitmapImage;

            Color color2 = (Color)ColorConverter.ConvertFromString(hexTxt);
            SolidColorBrush brush2 = new SolidColorBrush(color2);
            tbk.Foreground = brush2;
            tbk.Content = txt;

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
