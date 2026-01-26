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
using zdhsys.Control;

namespace zdhsys.UI
{
    /// <summary>
    /// UCDevice.xaml 的交互逻辑
    /// </summary>
    public partial class UCDevice : UserControl
    {
        public UCDevice()
        {
            InitializeComponent();

            InitUI();
        }

        private void InitUI()
        {
            tab_dev.Click_Class += Tab_dev_Click_Class;
            tab_dev.Click_Device += Tab_dev_Click_Device;
            tab_dev.Click_DeviceGroup += Tab_dev_Click_DeviceGroup;
        }
        /// <summary>
        /// 设备组管理
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Tab_dev_Click_DeviceGroup(object sender, RoutedEventArgs e)
        {
            DeviceGroupTabControl dgt = new DeviceGroupTabControl();
            dcc.Content = dgt;
        }
        /// <summary>
        /// 设备管理
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Tab_dev_Click_Device(object sender, RoutedEventArgs e)
        {
            DeviceTabControl dtc = new DeviceTabControl();
            dcc.Content = dtc;
        }
        /// <summary>
        /// 类别管理
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Tab_dev_Click_Class(object sender, RoutedEventArgs e)
        {
            DeviceClassTabControl dct = new DeviceClassTabControl();
            dcc.Content = dct;
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            //默认加载设备管理
            DeviceTabControl dtc = new DeviceTabControl();
            dcc.Content = dtc;
        }
    }
}
