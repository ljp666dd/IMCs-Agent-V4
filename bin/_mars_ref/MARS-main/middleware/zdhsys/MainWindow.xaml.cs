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
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using zdhsys.Control;
using zdhsys.UI;
using zdhsys.Unitils;
using zdhsys.viewModel;

namespace zdhsys
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            SqlHelper.InitSQLite();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            // 窗口移动事件
            //MouseDown += delegate { DragMove(); };
            //MouseLeftButtonDown += delegate { DragMove(); };
            MouseLeftButtonDown += UserControl_MouseLeftButtonDown;
            InitUI();
        }

        private void UserControl_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Left)
            {
                try
                {
                    Window window = Window.GetWindow(this);
                    window.DragMove();
                }
                catch { }
            }
        }

        private void InitUI()
        {
            Btn_home.Set_Name("主页");
            Btn_option.Set_Name("配方管理");
            Btn_flow.Set_Name("流程管理");
            Btn_device.Set_Name("设备管理");
            Btn_data.Set_Name("数据管理");

            btn_min.SetUI("\\Image\\矩形 105.png");
            btn_max.SetUI("\\Image\\窗口.png");
            btn_close.SetUI("\\Image\\语音关闭.png");
        }

        #region 缩小放大关闭按钮事件
        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            //强行退出
            Environment.Exit(0);
        }

        private void Btn_Max_Click(object sender, RoutedEventArgs e)
        {
            if(WindowState == WindowState.Normal)
            {
                WindowState = WindowState.Maximized;
            }
            else if(WindowState == WindowState.Maximized)
            {
                WindowState = WindowState.Normal;
            }
        }

        private void Btn_Min_Click(object sender, RoutedEventArgs e)
        {
            WindowState = WindowState.Minimized;
        }
        #endregion

        #region 左侧菜单栏相关
        double max = 200; //缓存左边列宽最大值
        double min = 70; // 左边列宽最小值
        // 左上角的菜单缩放按钮事件
        private void BtnMenu_Click(object sender, RoutedEventArgs e)
        {
            if (Gd_Left.ActualWidth > min)
            {
                // 变小
                GridLength fromValue = new GridLength(max); // 设置目标宽度
                GridLength toValue = new GridLength(min); // 设置目标宽度
                TimeSpan duration = TimeSpan.FromSeconds(0.2); // 设置动画持续时间
                AnimateGridColumnWidth(Gd, 0, fromValue, toValue, duration);
                SetButtonMaxMin(false);
            }
            else
            {
                // 变大
                TimeSpan duration = TimeSpan.FromSeconds(0.2); // 设置动画持续时间
                AnimateGridColumnWidth(Gd, 0, new GridLength(min), new GridLength(max), duration);
                SetButtonMaxMin(true);
            }
        }
        // 左边菜单动画效果
        public static void AnimateGridColumnWidth(Grid targetGrid, int columnIndex, GridLength fromValue, GridLength toValue, TimeSpan duration)
        {
            // 创建 GridLengthAnimation 对象并设置属性
            GLA animation = new GLA
            {
                From = fromValue,
                To = toValue,
                Duration = duration
            };
            // 将动画应用于 ColumnDefinition 的 Width 属性
            ColumnDefinition columnDefinition = targetGrid.ColumnDefinitions[columnIndex];
            columnDefinition.BeginAnimation(ColumnDefinition.WidthProperty, animation);
        }

        // 控制左边按钮大小变化
        private void SetButtonMaxMin(bool flag)
        {
            Btn_home.setCollapsed(flag);
            Btn_option.setCollapsed(flag);
            Btn_flow.setCollapsed(flag);
            Btn_device.setCollapsed(flag);
            Btn_data.setCollapsed(flag);
        }

        private UCHome home = new UCHome();

        // 菜单按钮单击事件
        private void Btn_home_Click(object sender, RoutedEventArgs e)
        {
            DevpButton devp = sender as DevpButton;
            //修改其它按钮的选中状态
            SetButtonCheck(devp.Name);
            // 菜单按钮对应的单击事件
            switch (devp.Name)
            {
                case "Btn_home":
                    dcc.Content = home;
                    break;
                case "Btn_option":
                    UCOption option = new UCOption();
                    dcc.Content = option;
                    break;
                case "Btn_flow":
                    UCFlow flow = new UCFlow();
                    dcc.Content = flow;
                    break;
                case "Btn_device":
                    UCDevice device = new UCDevice();
                    dcc.Content = device;
                    break;
                case "Btn_data":
                    UCData data = new UCData();
                    dcc.Content = data;
                    break;
                default:
                    break;
            }
        }
        // 设置其它按钮为未选中状态
        private void SetButtonCheck(string btnName)
        {
            if(btnName != "Btn_home")
            {
                Btn_home.setFocus(false);
            }

            if (btnName != "Btn_option")
            {
                Btn_option.setFocus(false);
            }
            if (btnName != "Btn_flow")
            {
                Btn_flow.setFocus(false);
            }
            if (btnName != "Btn_device")
            {
                Btn_device.setFocus(false);
            }

            if (btnName != "Btn_data")
            {
                Btn_data.setFocus(false);
            }
        }

        #endregion
    }
}
