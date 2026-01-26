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
using zdhsys.Popup;
using zdhsys.Unitils;

namespace zdhsys.UI
{
    /// <summary>
    /// OptionCreateControl.xaml 的交互逻辑
    /// </summary>
    public partial class OptionCreateControl : UserControl
    {
        public OptionCreateControl()
        {
            InitializeComponent();
            InitUI();
        }

        //返回上一个窗口
        public event RoutedEventHandler Click;

        private void InitUI()
        {
            Btn_Add.Click += Btn_Add_Click;
            Btn_Save.Click += Btn_Save_Click;
            Btn_Close.Click += Btn_Close_Click;
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            if (Click != null)
            {
                Click.Invoke(this, e);
            }
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            txt_option.SetUI("输入配方名称");

            Btn_Add.SetUI("#E8F4FF", "#97D4FF", "\\Image\\add.png", "#1990FF", "增加");
            Btn_Save.SetUI("#E8F4FF", "#97D4FF", "\\Image\\save.png", "#1990FF", "保存");
        }

        /// <summary>
        /// 保存当前配方
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {

        }
        /// <summary>
        /// 新增配方内容
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_Add_Click(object sender, RoutedEventArgs e)
        {
            Point p = GetParentWindows.GetPoint(this);
            AddOptionInfo info = new AddOptionInfo(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            _ = info.ShowDialog();
        }
    }
}
