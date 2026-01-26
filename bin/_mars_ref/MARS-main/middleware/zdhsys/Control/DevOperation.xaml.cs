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
    /// DevOperation.xaml 的交互逻辑
    /// </summary>
    public partial class DevOperation : UserControl
    {
        public DevOperation()
        {
            InitializeComponent();
        }
        public event RoutedEventHandler Click_Details;
        public event RoutedEventHandler Click_Update;
        public event RoutedEventHandler Click_Delete;
        public object obj;
        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            btn_details.SetUI("\\Image\\details.png", "#027AFF", "详情");
            btn_del.SetUI("\\Image\\delete.png", "#FF0000", "删除");
            btn_edit.SetUI("\\Image\\edit.png", "#027AFF", "修改");

            btn_details.Click += Btn_Click;
            btn_del.Click += Btn_Click;
            btn_edit.Click += Btn_Click;
        }

        private void Btn_Click(object sender, RoutedEventArgs e)
        {
            DevButton2 btn = sender as DevButton2;
            if(btn.Text == "详情")
            {
                if (Click_Details != null)
                {
                    Click_Details.Invoke(this, e);
                }
            }
            else if (btn.Text == "修改")
            {
                if (Click_Update != null)
                {
                    Click_Update.Invoke(this, e);
                }
            }
            else if (btn.Text == "删除")
            {
                if (Click_Delete != null)
                {
                    Click_Delete.Invoke(this, e);
                }
            }
        }
    }
}
