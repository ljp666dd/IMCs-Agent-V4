using System.Collections.Generic;
using System.Windows;
using zdhsys.Bean;
using zdhsys.Unitils;

namespace zdhsys.Popup
{
    /// <summary>
    /// AddOptionInfo.xaml 的交互逻辑
    /// </summary>
    public partial class AddSubOptionInfo : Window
    {
        public AddSubOptionInfo(Point p)
        {
            InitializeComponent();
            Left = p.X;
            Top = p.Y;
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private List<SubOptionModel> dim = null;
        public OptionFieldsModel2 ofm = null;
        public bool isUpdate = false;
        public bool flag_Save = false;

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            Btn_Save.SetUI("#0066FF", "#0066FF", "#FFFFFF", "载入");
            Btn_Close1.SetUI("#FFFFFF", "#0066FF", "#0066FF", "取消");

            dim = SqlHelper.GetSubOptionModel();
            if (dim == null)
            {
                return;
            }

            if (dim.Count == 0)
            {
                Point p = new Point(Left, Top);
                MessageInfo info = new MessageInfo(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                info.SetUI("请先添加子配方");
                _ = info.ShowDialog();
                Close();
            }
            int index = 0;
            for (int i = 0; i < dim.Count; i++)
            {
                cb.Items.Add(dim[i].SubOptionName);
                if (ofm != null && ofm.Id == dim[i].Id)
                {
                    index = i;
                }
            }
            cb.SelectedIndex = index;
        }

        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            if (!isUpdate)
            {
                ofm = new OptionFieldsModel2();
            }
            ofm.Id = dim[cb.SelectedIndex].Id;
            ofm.TagName = dim[cb.SelectedIndex].SubOptionName;
            ofm.IsSub = true;
            ofm.SubId = dim[cb.SelectedIndex].Id;
            flag_Save = true;
            Close();
        }
    }
}
