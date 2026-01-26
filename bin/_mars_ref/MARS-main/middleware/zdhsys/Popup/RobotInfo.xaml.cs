using System;
using System.Collections.Generic;
using System.Windows;
using zdhsys.Bean;
using zdhsys.Unitils;

namespace zdhsys.Popup
{
    /// <summary>
    /// AddOptionInfo.xaml 的交互逻辑
    /// </summary>
    public partial class RobotInfo : Window
    {
        public RobotInfo(Point p)
        {
            InitializeComponent();
            Left = p.X;
            Top = p.Y;
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            //只能输入数字
            txt_Port.SetNumber();
            txt_Port.SetUI("输入数值");
            txt_IP.SetUI("输入IP地址");

            Btn_Connect.SetUI("#0066FF", "#0066FF", "#FFFFFF", "测试连接");
            Btn_Save.SetUI("#0066FF", "#0066FF", "#FFFFFF", "保存");
            Btn_Close1.SetUI("#FFFFFF", "#0066FF", "#0066FF", "取消");

            InitUI();
        }

        private List<RobotInfoModel> vs = null;

        private void InitUI()
        {
            //加载数据进来，目前机械臂只有一个
            vs = SqlHelper.GetRobotInfo();
            if (vs.Count > 0)
            {
                txt_Port.SetText(vs[0].RobotPort);
                txt_IP.SetText(vs[0].RobotIP);
            }
        }

        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            RobotInfoModel robot = null;
            if (vs.Count > 0)
            {
                robot = vs[0];
            }
            else if (vs.Count == 0)
            {
                robot = new RobotInfoModel();
            }
            robot.RobotIP = txt_IP.Text();
            robot.RobotPort = txt_Port.Text();
            int ret = 0;
            if (!int.TryParse(robot.RobotPort, out ret))
            {
                Point p = new Point(Left, Top);
                MessageInfo info = new MessageInfo(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                info.SetUI("端口不是整型");
                _ = info.ShowDialog();
                return;
            }
            if (vs.Count == 0)
            {
                SqlHelper.Add(robot);
            }
            else
            {
                SqlHelper.Update(robot);
            }
            Close();
        }

        private void Btn_Connect_Click(object sender, RoutedEventArgs e)
        {
            Point p = new Point(Left, Top);
            MessageInfo info = new MessageInfo(p);
            if (string.IsNullOrEmpty(txt_IP.Text()) || string.IsNullOrEmpty(txt_Port.Text()))
            {
                info.SetUI("IP和端口不能为空!!");
                _ = info.ShowDialog();
                return;
            }
            if (RoborUnitils.ConnectRobot(txt_IP.Text(), int.Parse(txt_Port.Text())))
            {
                info.SetUI("TCP连接成功!!");
                _ = info.ShowDialog();
            }
            //RoborUnitils.robot.Send("welcome robot");
        }
    }
}
