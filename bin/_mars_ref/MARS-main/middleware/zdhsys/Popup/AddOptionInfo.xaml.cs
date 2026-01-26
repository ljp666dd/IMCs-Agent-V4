using System.Collections.Generic;
using System.Windows;
using zdhsys.Bean;
using zdhsys.Unitils;

namespace zdhsys.Popup
{
    /// <summary>
    /// AddOptionInfo.xaml 的交互逻辑
    /// </summary>
    public partial class AddOptionInfo : Window
    {
        public AddOptionInfo(Point p)
        {
            InitializeComponent();
            Left = p.X;
            Top = p.Y;
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private List<DeviceInfoModel> dim = null;
        public OptionFieldsModel ofm = null;
        public bool isUpdate = false;
        public bool flag_Save = false;

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            //只能输入数字
            txt_num.SetNumber();
            txt_num.SetUI("输入数值");

            Btn_Save.SetUI("#0066FF", "#0066FF","#FFFFFF","载入");
            Btn_Close1.SetUI("#FFFFFF", "#0066FF", "#0066FF","取消");

            dim = SqlHelper.GetDeviceInfo();
            if (dim == null) return;
            if (dim.Count == 0)
            {
                Point p = new Point(Left,Top);
                MessageInfo info = new MessageInfo(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                info.SetUI("请先添加设备");
                _ = info.ShowDialog();
                Close();
            }
            int index = 0;
            for (int i = 0; i < dim.Count; i++)
            {
                cb.Items.Add(dim[i].TagName);
                if (ofm != null && ofm.Id == dim[i].Id)
                {
                    index = i;
                }
            }
            cb.SelectedIndex = index;
            if (ofm != null)
            {
                txt_num.SetText(ofm.TagValue);
            }
        }

        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(txt_num.Text()))
            {
                Point p = new Point(Left, Top);
                MessageInfo info = new MessageInfo(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                info.SetUI("数值不能为空");
                _ = info.ShowDialog();
                return;
            }
            if (!isUpdate)
            {
                ofm = new OptionFieldsModel();
            }
            ofm.Id = dim[cb.SelectedIndex].Id;
            ofm.DeviceId = dim[cb.SelectedIndex].DeviceId;
            ofm.TagName = dim[cb.SelectedIndex].TagName;
            ofm.TagValue = txt_num.Text();
            flag_Save = true;
            Close();
        }

        private void cb_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (dim != null && dim.Count > 0)
            {
                string unit = dim[cb.SelectedIndex].TagUnit;
                unit = "单位:" + unit;
                lb.Content = unit;
            }
        }
    }
}
