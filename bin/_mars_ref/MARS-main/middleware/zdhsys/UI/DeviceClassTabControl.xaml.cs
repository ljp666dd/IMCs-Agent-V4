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

namespace zdhsys.UI
{
    /// <summary>
    /// DeviceClassTabControl.xaml 的交互逻辑
    /// </summary>
    public partial class DeviceClassTabControl : UserControl
    {
        public DeviceClassTabControl()
        {
            InitializeComponent();
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            txt_option.SetUI("类别名称");

            btn_search.SetUI("#027AFF", "#027AFF", "\\Image\\搜索.png", "#FFFFFF", "搜索");
            btn_clear.SetUI("#FFFFFF", "#027AFF", "\\Image\\重置.png", "#027AFF", "重置");

            btn_add.SetUI("#E8F4FF", "#97D4FF", "\\Image\\add.png", "#1990FF", "创建");
            btn_del.SetUI("#F2F7FF", "#FF0000", "\\Image\\delete.png", "#FF0000", "批量删除");
        }
    }
}
