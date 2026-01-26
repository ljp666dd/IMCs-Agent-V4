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
using System.Windows.Shapes;
using zdhsys.entity;

namespace zdhsys.Popup
{
    /// <summary>
    /// AddDevicePoint.xaml 的交互逻辑
    /// </summary>
    public partial class AddDevicePoint : Window
    {
        public AddDevicePoint(Point p)
        {
            InitializeComponent();
            Left = p.X;
            Top = p.Y;
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        public DevicePoint dp;
        public bool isUpdate = false;
        public bool flag_Save = false;

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            txt_1.SetUI("输入坐标");
            txt_1.SetNumber();

            txt_2.SetUI("输入坐标");
            txt_2.SetNumber();

            txt_3.SetUI("输入坐标");
            txt_3.SetNumber();

            txt_4.SetUI("输入坐标");
            txt_4.SetNumber();

            txt_5.SetUI("输入坐标");
            txt_5.SetNumber();

            txt_6.SetUI("输入坐标");
            txt_6.SetNumber();

            txt_7.SetUI("输入坐标");
            txt_7.SetNumber();

            txt_8.SetUI("输入坐标");
            txt_8.SetNumber();

            txt_9.SetUI("输入坐标");
            txt_9.SetNumber();

            txt_10.SetUI("输入坐标");
            txt_10.SetNumber();

            txt_11.SetUI("输入坐标");
            txt_11.SetNumber();

            txt_12.SetUI("输入坐标");
            txt_12.SetNumber();

            Btn_Save.SetUI("#0066FF", "#0066FF", "#FFFFFF", "保存");
            Btn_Close1.SetUI("#FFFFFF", "#0066FF", "#0066FF", "取消");

            if (isUpdate)
            {
                txt_1.SetText(dp.p1);
                txt_2.SetText(dp.p2);
                txt_3.SetText(dp.p3);
                txt_4.SetText(dp.p4);
                txt_5.SetText(dp.p5);
                txt_6.SetText(dp.p6);
                txt_7.SetText(dp.p7);
                txt_8.SetText(dp.p8);
                txt_9.SetText(dp.p9);
                txt_10.SetText(dp.p10);
                txt_11.SetText(dp.p11);
                txt_12.SetText(dp.p12);
            }
        }

        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            if (!isUpdate)
            {
                dp = new DevicePoint();
            }
            dp.p1 = txt_1.Text();
            dp.p2 = txt_2.Text();
            dp.p3 = txt_3.Text();
            dp.p4 = txt_4.Text();
            dp.p5 = txt_5.Text();
            dp.p6 = txt_6.Text();
            dp.p7 = txt_7.Text();
            dp.p8 = txt_8.Text();
            dp.p9 = txt_9.Text();
            dp.p10 = txt_10.Text();
            dp.p11 = txt_11.Text();
            dp.p12 = txt_12.Text();
            flag_Save = true;
            Close();
        }
    }
}
