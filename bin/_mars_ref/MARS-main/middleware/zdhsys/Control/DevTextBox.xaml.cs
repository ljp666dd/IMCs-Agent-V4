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
    /// DevTextBox.xaml 的交互逻辑
    /// </summary>
    public partial class DevTextBox : UserControl
    {
        public DevTextBox()
        {
            InitializeComponent();
        }
        private string Tips = "";
        public void SetUI(string tips)
        {
            textBox.Text = tips;
            Tips = tips;
        }

        public string Text()
        {
            if (textBox.Text == Tips) return "";
            return textBox.Text;
        }

        public void SetText(string text)
        {
            textBox.Text = text;
        }

        /// <summary>
        /// 设置只能输入数字
        /// </summary>
        public void SetNumber()
        {
            textBox.PreviewKeyDown += TextBox_PreviewKeyDown;
        }

        private void TextBox_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            // 只允许输入数字和一些特殊按键
            if (!((Key.D0 <= e.Key && e.Key <= Key.D9) ||    // 数字键盘上的数字
                  (Key.NumPad0 <= e.Key && e.Key <= Key.NumPad9) ||  // 主键盘上的数字
                  e.Key == Key.Back ||     // 退格键
                  e.Key == Key.Delete ||   // 删除键
                  e.Key == Key.Tab ||      // Tab 键
                  e.Key == Key.Left || e.Key == Key.Right))  // 左右方向键
            {
                e.Handled = true; // 拦截非数字键
            }
        }

        private void TextBox_GotFocus(object sender, RoutedEventArgs e)
        {
            if (textBox.Text == Tips)
            {
                textBox.Text = "";
                textBox.Foreground = Brushes.Black;
            }
            else if (string.IsNullOrEmpty(textBox.Text))
            {
                textBox.Text = Tips;
                textBox.Foreground = Brushes.Gray;
            }
        }
    }
}
