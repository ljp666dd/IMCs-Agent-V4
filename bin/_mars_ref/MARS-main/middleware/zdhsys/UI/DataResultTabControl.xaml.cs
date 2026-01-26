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
    /// DataResultTabControl.xaml 的交互逻辑
    /// </summary>
    public partial class DataResultTabControl : UserControl
    {
        public DataResultTabControl()
        {
            InitializeComponent();
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            txt_option.SetUI("配方名称");
            txt_density.SetUI("输入密度值");
            txt_PCR.SetUI("输入数值");


            btn_search.SetUI("#027AFF", "#027AFF", "\\Image\\搜索.png", "#FFFFFF", "搜索");
            btn_clear.SetUI("#FFFFFF", "#027AFF", "\\Image\\重置.png", "#027AFF", "重置");

            btn_add.SetUI("#E8F4FF", "#97D4FF", "\\Image\\add.png", "#1990FF", "导出");
        }
    }
}
