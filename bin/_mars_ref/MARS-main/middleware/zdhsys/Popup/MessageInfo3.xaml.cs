using System.Windows;

namespace zdhsys.Popup
{
    /// <summary>
    /// AddOptionInfo.xaml 的交互逻辑
    /// </summary>
    public partial class MessageInfo3 : Window
    {
        public MessageInfo3(Point p)
        {
            InitializeComponent();
            Left = p.X;
            Top = p.Y;
        }

        public void SetUI(string str)
        {
            lb.Content = str;
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            Btn_Save.SetUI("#0066FF", "#0066FF", "#FFFFFF", "继续执行");
            Btn_Over.SetUI("#0066FF", "#0066FF", "#FFFFFF", "结束执行");
        }

        public bool flag = false;

        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            flag = true;
            Close();
        }

        private void Btn_Over_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}
