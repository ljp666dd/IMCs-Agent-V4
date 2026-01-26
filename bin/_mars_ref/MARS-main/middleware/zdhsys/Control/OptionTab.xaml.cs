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
    /// OptionTab.xaml 的交互逻辑
    /// </summary>
    public partial class OptionTab : UserControl
    {
        public OptionTab()
        {
            InitializeComponent();
        }

        public event RoutedEventHandler Click_Option;
        public event RoutedEventHandler Click_SubOption;

        public string str_1 = "小瓶";
        public string str_2 = "大瓶";

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            btn_Option.SetUI(str_1);
            btn_Option.Click += Btn_Click;
            btn_Option.SetCheck(true);


            btn_SubOption.SetUI(str_2);
            btn_SubOption.Click += Btn_Click;
            btn_SubOption.SetCheck(false);
        }

        private void Btn_Click(object sender, RoutedEventArgs e)
        {
            TabButton tab = sender as TabButton;
            if (tab.Text == str_1)
            {
                btn_Option.SetCheck(true);
                btn_SubOption.SetCheck(false);
                if (Click_Option != null)
                {
                    Click_Option.Invoke(this, e);
                }
            }
            else if (tab.Text == str_2)
            {
                btn_Option.SetCheck(false);
                btn_SubOption.SetCheck(true);
                if (Click_SubOption != null)
                {
                    Click_SubOption.Invoke(this, e);
                }
            }
        }
    }
}
