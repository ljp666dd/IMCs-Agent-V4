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
    /// UCOption.xaml 的交互逻辑
    /// </summary>
    public partial class UCOption : UserControl
    {
        public UCOption()
        {
            InitializeComponent();

            tab_opt.Click_Option += Tab_opt_Click_Option;
            tab_opt.Click_SubOption += Tab_opt_Click_SubOption;
        }

        private void Tab_opt_Click_SubOption(object sender, RoutedEventArgs e)
        {
            SubOptionTab tab = new SubOptionTab();
            dcc.Content = tab;
        }

        private void Tab_opt_Click_Option(object sender, RoutedEventArgs e)
        {
            OptionTab tab = new OptionTab();
            dcc.Content = tab;
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            OptionTab tab = new OptionTab();
            dcc.Content = tab;
        }
    }
}
