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
    /// TabButton.xaml 的交互逻辑
    /// </summary>
    public partial class TabButton : UserControl
    {
        public TabButton()
        {
            InitializeComponent();
        }

        public event RoutedEventHandler Click;

        public string Text;
        public void SetUI(string content)
        {
            btn.Content = content;
            Text = content;
        }

        /// <summary>
        /// 设备选中状态
        /// </summary>
        /// <param name="flag"></param>
        public void SetCheck(bool flag)
        {
            if (flag)
            {
                Color color = (Color)ColorConverter.ConvertFromString("#333333");
                SolidColorBrush brush = new SolidColorBrush(color);
                btn.Foreground = brush;
                bd.Visibility = Visibility.Visible;
            }
            else
            {
                Color color = (Color)ColorConverter.ConvertFromString("#999999");
                SolidColorBrush brush = new SolidColorBrush(color);
                btn.Foreground = brush;
                bd.Visibility = Visibility.Hidden;
            }
        }

        private void btn_Click(object sender, RoutedEventArgs e)
        {
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
        }
    }
}
