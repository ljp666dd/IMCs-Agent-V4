using System.Windows;

namespace zdhsys.Popup
{
    /// <summary>
    /// AddDeviceFields.xaml 的交互逻辑
    /// </summary>
    public partial class AddDeviceFields : Window
    {
        public AddDeviceFields(Point p)
        {
            InitializeComponent();
            Left = p.X;
            Top = p.Y;
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            Btn_Save.SetUI("#0066FF", "#0066FF", "#FFFFFF", "确定");
            Btn_Close1.SetUI("#FFFFFF", "#0066FF", "#0066FF", "取消");

            txt_fields.SetUI("输入字段名称");
            txt_content.SetUI("输入内容");
            if (!string.IsNullOrEmpty(fieldsName))
            {
                txt_fields.SetText(fieldsName);
            }
            if (!string.IsNullOrEmpty(fieldsContent))
            {
                txt_content.SetText(fieldsContent);
            }
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        public bool flag_save = false;
        public string fieldsName = "";
        public string fieldsContent = "";

        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            fieldsName = txt_fields.Text();
            fieldsContent = txt_content.Text();
            if (string.IsNullOrEmpty(fieldsName) || string.IsNullOrEmpty(fieldsContent))
            {
                return;
            }
            flag_save = true;
            Close();
        }
    }
}
