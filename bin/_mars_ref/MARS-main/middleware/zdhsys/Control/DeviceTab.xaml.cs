using System.Windows;
using System.Windows.Controls;

namespace zdhsys.Control
{
    /// <summary>
    /// DeviceTab.xaml 的交互逻辑
    /// </summary>
    public partial class DeviceTab : UserControl
    {
        public DeviceTab()
        {
            InitializeComponent();
        }

        public event RoutedEventHandler Click_Device;
        public event RoutedEventHandler Click_DeviceGroup;
        public event RoutedEventHandler Click_Class;

        public string str_1 = "设备管理";
        public string str_2 = "设备组管理";
        public string str_3 = "类别管理";

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            btn_device.SetUI(str_1);
            btn_device.Click += Btn_Click;
            btn_device.SetCheck(true);


            btn_device_group.SetUI(str_2);
            btn_device_group.Click += Btn_Click;
            btn_device_group.SetCheck(false);

            btn_class.SetUI(str_3);
            btn_class.Click += Btn_Click;
            btn_class.SetCheck(false);
        }

        private void Btn_Click(object sender, RoutedEventArgs e)
        {
            TabButton tab = sender as TabButton;
            if (tab.Text == str_1)
            {
                btn_device.SetCheck(true);
                btn_device_group.SetCheck(false);
                btn_class.SetCheck(false);
                if (Click_Device != null)
                {
                    Click_Device.Invoke(this, e);
                }
            }
            else if (tab.Text == str_2)
            {
                btn_device.SetCheck(false);
                btn_device_group.SetCheck(true);
                btn_class.SetCheck(false);
                if (Click_DeviceGroup != null)
                {
                    Click_DeviceGroup.Invoke(this, e);
                }
            }
            else if (tab.Text == str_3)
            {
                btn_device.SetCheck(false);
                btn_device_group.SetCheck(false);
                btn_class.SetCheck(true);
                if (Click_Class != null)
                {
                    Click_Class.Invoke(this, e);
                }
            }
        }
    }
}
