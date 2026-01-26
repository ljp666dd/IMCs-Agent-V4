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
    /// DevpButton.xaml 的交互逻辑
    /// </summary>
    public partial class DevpButton : UserControl
    {
        public DevpButton()
        {
            InitializeComponent();
        }

        private bool flag = false; // 是否选中
        private bool max = true;  // 是大，还是小
        // 按钮文本
        public string btnName = "主页";

        Image img;
        Image img2;
        TextBlock tbk;
        // 把单击事件传出去。
        public event RoutedEventHandler Click;

        // 配色
        string hex1 = "#98A3FE";
        string hex2 = "#FFFFFF";
        string hex3 = "#465785";
        private static string basePath = "\\Image\\menu\\";
        // 白色ICO
        string imgPath = basePath + "主页.png";
        // 黑色ICO
        string imgPath2 = basePath + "主页2.png";

        // 封装主菜单所有ICO切换
        public void Set_Name(string name)
        {
            InitUI();
            btnName = name;
            switch (btnName)
            {
                case "主页":
                    imgPath = basePath + "主页.png";
                    imgPath2 = basePath + "主页2.png";
                    break;
                case "配方管理":
                    imgPath = basePath + "配方.png";
                    imgPath2 = basePath + "配方2.png";
                    break;
                case "流程管理":
                    imgPath = basePath + "流程.png";
                    imgPath2 = basePath + "流程2.png";
                    break;
                case "设备管理":
                    imgPath = basePath + "设备.png";
                    imgPath2 = basePath + "设备2.png";
                    break;
                case "数据管理":
                    imgPath = basePath + "数据.png";
                    imgPath2 = basePath + "数据2.png";
                    break;
            }
            updateUI(hex2, hex3, imgPath2);
        }


        // 大小按钮单击事件
        private void btn_Click(object sender, RoutedEventArgs e)
        {
            flag = true;
            if (flag)
            {
                if (max)
                    updateUI(hex1, hex2, imgPath);
                else
                    updateUI2(hex1, imgPath);
            }
            else
            {
                if (max)
                    updateUI(hex2, hex3, imgPath2);
                else
                    updateUI2(hex2, imgPath2);
            }
            if(Click != null)
                Click.Invoke(this, e);
        }

        // 更新大按钮界面
        private void updateUI(string hexValue, string hex, string path)
        {
            Color color = (Color)ColorConverter.ConvertFromString(hexValue);
            SolidColorBrush brush = new SolidColorBrush(color);
            btn.Background = brush;
            Uri imageUri = new Uri(AppDomain.CurrentDomain.BaseDirectory + path);
            BitmapImage bitmapImage = new BitmapImage(imageUri);
            img.Source = bitmapImage;
            Color color2 = (Color)ColorConverter.ConvertFromString(hex);
            SolidColorBrush brush2 = new SolidColorBrush(color2);
            tbk.Foreground = brush2;
            tbk.Text = btnName;
            // 这里加长一点，与其它4个中文的文本长度对齐。
            if(btnName == "主页")
            {
                tbk.Text = "主页" + '\u00A0' + '\u00A0' + '\u00A0' + '\u00A0' + '\u00A0' + '\u00A0' + '\u00A0';
            }
            tbk.FontSize = 15;
            tbk.FontFamily = new FontFamily("Microsoft YaHei");
            tbk.FontWeight = FontWeights.Bold;
        }
        // 更新小按钮界面
        private void updateUI2(string hexValue, string path)
        {
            Color color = (Color)ColorConverter.ConvertFromString(hexValue);
            SolidColorBrush brush = new SolidColorBrush(color);
            btn2.Background = brush;
            Uri imageUri = new Uri(AppDomain.CurrentDomain.BaseDirectory + path);
            BitmapImage bitmapImage = new BitmapImage(imageUri);
            img2.Source = bitmapImage;
        }

        // 取消选中状态
        public void setFocus(bool check)
        {
            flag = check;
            if (flag)
            {
                if (max)
                    updateUI(hex1, hex2, imgPath);
                else
                    updateUI2(hex1, imgPath);
            }
            else
            {
                if (max)
                    updateUI(hex2, hex3, imgPath2);
                else
                    updateUI2(hex2, imgPath2);
            }
        }

        // 设置大小状态
        public void setCollapsed(bool see)
        {
            if (see)
            {
                this.Width = 150;
                btn.Visibility = Visibility.Visible;
                btn2.Visibility = Visibility.Collapsed;
                max = true;
            }
            else
            {
                this.Width = 25;
                btn.Visibility = Visibility.Collapsed;
                btn2.Visibility = Visibility.Visible;
                max = false;
            }
            setFocus(flag);
        }

        private void InitUI()
        {
            // 找出控件里面子控件
            img = btn.Template.FindName("img", btn) as Image;
            img2 = btn2.Template.FindName("img2", btn2) as Image;
            tbk = btn.Template.FindName("tbk", btn) as TextBlock;

            //小状态的按钮，先隐藏。  这里注意：如果在代码里面写了隐藏，上面的img2就会找不到。为null
            //只能先获取对象，后做隐藏处理。
            btn2.Visibility = Visibility.Collapsed;

        }
    }
}
