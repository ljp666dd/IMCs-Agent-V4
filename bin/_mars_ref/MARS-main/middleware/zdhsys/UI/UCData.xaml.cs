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
    /// UCData.xaml 的交互逻辑
    /// </summary>
    public partial class UCData : UserControl
    {
        public UCData()
        {
            InitializeComponent();
            InitUI();
        }

        private void InitUI()
        {
            tab_dev.str_1 = "结果数据";
            tab_dev.str_2 = "设备数据";
            tab_dev.str_3 = "流程数据";

            tab_dev.Click_Class += Tab_dev_Click_Class;
            tab_dev.Click_Device += Tab_dev_Click_Device;
            tab_dev.Click_DeviceGroup += Tab_dev_Click_DeviceGroup;
        }

        private void Tab_dev_Click_DeviceGroup(object sender, RoutedEventArgs e)
        {
            DataDeviceTabControl ddt = new DataDeviceTabControl();
            dcc.Content = ddt;
        }

        private void Tab_dev_Click_Device(object sender, RoutedEventArgs e)
        {
            DataResultTabControl drt = new DataResultTabControl();
            dcc.Content = drt;
        }

        private void Tab_dev_Click_Class(object sender, RoutedEventArgs e)
        {
            DataFlowTabControl dft = new DataFlowTabControl();
            dcc.Content = dft;
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            DataResultTabControl drt = new DataResultTabControl();
            dcc.Content = drt;
        }
    }
}
