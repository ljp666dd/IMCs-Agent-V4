using Newtonsoft.Json;
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
using System.Windows.Shapes;
using zdhsys.Bean;
using zdhsys.Control;
using zdhsys.Unitils;

namespace zdhsys.Popup
{
    /// <summary>
    /// AddOption.xaml 的交互逻辑
    /// </summary>
    public partial class AddOption : Window
    {
        public AddOption(Point p)
        {
            InitializeComponent();

            Left = p.X;
            Top = p.Y;

            Btn_Add.Click += Btn_Add_Click;
            Btn_Add2.Click += Btn_Add2_Click;
            Btn_Close.Click += Btn_Close_Click;
            Btn_Save.Click += Btn_Save_Click;
        }

        private List<OptionFieldsModel2> vs = new List<OptionFieldsModel2>();
        public bool isUpdate = false;
        public bool flag_Save = false;
        public Options opt = null;

        private void Btn_Add2_Click(object sender, RoutedEventArgs e)
        {
            Point p = new Point(Left, Top);
            AddSubOptionInfo info = new AddSubOptionInfo(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            _ = info.ShowDialog();
            if (info.flag_Save)
            {
                vs.Add(info.ofm);
                DataToUI();
            }
        }

        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            Point p = new Point(Left, Top);
            MessageInfo info = new MessageInfo(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            //做一些过滤
            if (string.IsNullOrEmpty(txt_option.Text()))
            {
                info.SetUI("配方名称不能为空");
                _ = info.ShowDialog();
                return;
            }
            if (vs.Count == 0)
            {
                info.SetUI("配方内容不能为空");
                _ = info.ShowDialog();
                return;
            }

            if (!isUpdate)
            {
                opt = new Options
                {
                    CreateTime = GlobalUnitils.GetNowTime(DateTime.Now)
                };
            }
            opt.OptionJson = JsonConvert.SerializeObject(vs);
            opt.OptionName = txt_option.Text();
            opt.Status = cb.SelectedIndex;
            flag_Save = true;
            Close();
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private void Btn_Add_Click(object sender, RoutedEventArgs e)
        {
            Point p = new Point(Left, Top);
            AddOptionInfo info = new AddOptionInfo(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            _ = info.ShowDialog();
            if (info.flag_Save)
            {
                OptionFieldsModel2 ofm2 = new OptionFieldsModel2
                {
                    DeviceId = info.ofm.DeviceId,
                    Id = info.ofm.Id,
                    IsSub = false,
                    TagName = info.ofm.TagName,
                    TagValue = info.ofm.TagValue
                };
                vs.Add(ofm2);
                DataToUI();
            }
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            txt_option.SetUI("输入配方名称");

            Btn_Add.SetUI("#E8F4FF", "#97D4FF", "\\Image\\add.png", "#1990FF", "增加");
            Btn_Add2.SetUI("#E8F4FF", "#97D4FF", "\\Image\\add.png", "#1990FF", "增加子配方");
            Btn_Save.SetUI("#E8F4FF", "#97D4FF", "\\Image\\save.png", "#1990FF", "保存");
            Btn_Close.SetUI("#E8F4FF", "#97D4FF", "\\Image\\语音关闭.png", "#1990FF", "关闭");

            cb.Items.Add("正常");
            cb.Items.Add("作废");
            cb.SelectedIndex = 0;

            if (isUpdate)
            {
                cb.SelectedIndex = opt.Status;
                txt_option.SetText(opt.OptionName);
                vs = JsonConvert.DeserializeObject<List<OptionFieldsModel2>>(opt.OptionJson);
                DataToUI();
            }
        }

        private void DataToUI()
        {
            gd_content.Children.Clear();
            gd_content.RowDefinitions.Clear();
            gd_content.ColumnDefinitions.Clear();

            for (int i = 0; i < vs.Count; i++)
            {
                DataToGridView(vs[i]);
            }
        }

        private void DataToGridView(OptionFieldsModel2 df)
        {
            // 创建新的行- 固定行高:40
            GridViewUnitls.AddRow(gd_content);

            // 创建并添加新的列到新行
            for (int i = 0; i < gd_header.ColumnDefinitions.Count; i++)
            {
                //当前行，添加1列 -- 列宽对应表头
                GridViewUnitls.AddColumn(gd_content, gd_header, i);

                // 添加对应的控件到新的格子中
                if (i == 0)
                {
                    GridViewUnitls.AddCell(gd_content, i, df.TagName);
                }
                else if (i == 1)
                {
                    GridViewUnitls.AddCell(gd_content, i, df.DeviceId);
                }
                else if (i == 2)
                {
                    GridViewUnitls.AddCell(gd_content, i, df.TagValue);
                }
                else if (i == 3)
                {
                    GridViewUnitls.AddCell(gd_content, i, df.IsSub ? "是" : "否");
                }
                else if (i == 4)
                {
                    DevOperation2 dev = new DevOperation2
                    {
                        HorizontalAlignment = HorizontalAlignment.Center,
                        VerticalAlignment = VerticalAlignment.Center,
                        obj = df//绑定对象。用于详情，修改，删除之类的。
                    };
                    dev.Click_Update += Dev_Click_Update;
                    dev.Click_Delete += Dev_Click_Delete;
                    Grid.SetRow(dev, gd_content.RowDefinitions.Count - 1);
                    Grid.SetColumn(dev, i);
                    gd_content.Children.Add(dev);
                }
            }
            //每行再加一个分割线
            GridViewUnitls.AddSpLine(gd_content, gd_header);
        }

        private void Dev_Click_Delete(object sender, RoutedEventArgs e)
        {
            DevOperation2 dev = sender as DevOperation2;
            OptionFieldsModel2 df = dev.obj as OptionFieldsModel2;
            for (int i = 0; i < vs.Count; i++)
            {
                if (vs[i].Id == df.Id)
                {
                    vs.RemoveAt(i);
                    break;
                }
            }
            DataToUI();
        }

        private void Dev_Click_Update(object sender, RoutedEventArgs e)
        {
            DevOperation2 dev = sender as DevOperation2;
            OptionFieldsModel2 df = dev.obj as OptionFieldsModel2;
            int index = -1;
            for (int i = 0; i < vs.Count; i++)
            {
                if (vs[i].Id == df.Id)
                {
                    index = i;
                    break;
                }
            }

            Point p = new Point(Left, Top);
            if (df.IsSub)
            {
                AddSubOptionInfo info = new AddSubOptionInfo(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                info.isUpdate = true;
                info.ofm = df;
                _ = info.ShowDialog();
                if (info.flag_Save)
                {
                    vs[index] = info.ofm;
                    DataToUI();
                }
            }
            else
            {
                AddOptionInfo info = new AddOptionInfo(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                info.isUpdate = true;
                OptionFieldsModel ofm = new OptionFieldsModel
                {
                    DeviceId = df.DeviceId,
                    Id = df.Id,
                    TagName = df.TagName,
                    TagValue = df.TagValue
                };
                info.ofm = ofm;
                _ = info.ShowDialog();
                if (info.flag_Save)
                {
                    OptionFieldsModel2 ofm2 = new OptionFieldsModel2
                    {
                        DeviceId = info.ofm.DeviceId,
                        Id = info.ofm.Id,
                        IsSub = false,
                        TagName = info.ofm.TagName,
                        TagValue = info.ofm.TagValue
                    };
                    vs[index] = ofm2;
                    DataToUI();
                }
            }
        }
    }
}
