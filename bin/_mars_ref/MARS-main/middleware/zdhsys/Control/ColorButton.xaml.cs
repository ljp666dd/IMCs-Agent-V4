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
    /// ColorButton.xaml 的交互逻辑
    /// </summary>
    public partial class ColorButton : UserControl
    {
        public ColorButton()
        {
            InitializeComponent();
        }
        public event RoutedEventHandler Click;

        public bool isCheck = false;
        public void SetUI(string hexColor)
        {
            Color color = (Color)ColorConverter.ConvertFromString(hexColor);
            SolidColorBrush brush = new SolidColorBrush(color);
            gd.Background = brush;
        }
        public void SetIsCheck(bool flag)
        {
            isCheck = flag;
            lb_gou.Visibility = isCheck ? Visibility.Visible : Visibility.Hidden;
        }

        private void UserControl_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            //isCheck = !isCheck;
            //lb_gou.Visibility = isCheck ? Visibility.Visible : Visibility.Hidden;
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
        }
    }
}
