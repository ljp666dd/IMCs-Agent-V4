using System.Windows;

namespace zdhsys.Popup
{
    /// <summary>
    /// AddOptionInfo.xaml 的交互逻辑
    /// </summary>
    public partial class DeleteInfo : Window
    {
        public DeleteInfo(Point p)
        {
            InitializeComponent();
            InitUI();
            Left = p.X;
            Top = p.Y;
        }
        private void InitUI()
        {

        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        public void SetUI(string str)
        {
            lb.Content = "是否删除当前" + str + "?";
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            Btn_Save.SetUI("#0066FF", "#0066FF", "#FFFFFF", "确定");
            Btn_Close1.SetUI("#FFFFFF", "#0066FF", "#0066FF", "取消");
        }

        public bool flagDelete = false;

        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            flagDelete = true;
            Close();
        }
    }
}
