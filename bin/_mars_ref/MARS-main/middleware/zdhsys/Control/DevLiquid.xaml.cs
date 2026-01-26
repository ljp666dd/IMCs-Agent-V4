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
    /// DevLiquid.xaml 的交互逻辑
    /// </summary>
    public partial class DevLiquid : UserControl
    {
        public DevLiquid()
        {
            InitializeComponent();
        }

        public void SetUI(int num, string path, string name, string status, string temp, string time,string daogui)
        {
            double wid = ActualWidth;
            double wids = wid / (num + 1);
            Console.WriteLine("wids=" + wids);
            for (int i = 0; i < num; i++)
            {
                DevDevice newButton = new DevDevice();
                int index = i + 1;
                newButton.SetUI("设备" + index, path);
                // 在 Grid 的列定义中增加一个新列
                //gd.ColumnDefinitions.Add(new ColumnDefinition() { Width = GridLength.Auto });
                gd.ColumnDefinitions.Add(new ColumnDefinition() { Width = new GridLength(wids) });

                // 将新按钮添加到 Grid 的新列
                Grid.SetColumn(newButton, i);
                gd.Children.Add(newButton);
            }

            lbName.Content = name;
            lbStatus.Content = status;
            lbTemp.Content = !string.IsNullOrEmpty(temp) ? temp + "℃" : "";
            lbTime.Content = time;
            lbDaogui.Content = daogui;
        }
    }
}
