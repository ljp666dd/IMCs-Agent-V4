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
    public partial class AddFlowData : Window
    {
        public AddFlowData(Point p)
        {
            InitializeComponent();

            Left = p.X;
            Top = p.Y;

            Btn_Add.Click += Btn_Add_Click;
            Btn_Close.Click += Btn_Close_Click;
            Btn_Save.Click += Btn_Save_Click;
        }

        public FlowModel fm = null;
        public bool isUpdate = false;
        public bool flag_Save = false;

        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            Point p = new Point(Left, Top);
            MessageInfo info = new MessageInfo(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            if (string.IsNullOrEmpty(txt_option.Text()))
            {
                info.SetUI("流程名称不能为空");
                _ = info.ShowDialog();
                return;
            }
            if (string.IsNullOrEmpty(txt_no.Text()))
            {
                info.SetUI("流程编号不能为空");
                _ = info.ShowDialog();
                return;
            }
            if (vs.Count == 0)
            {
                info.SetUI("流程内容不能为空");
                _ = info.ShowDialog();
                return;
            }

            if (isUpdate)
            {
                fm.FlowName = txt_option.Text();
                fm.FlowNO = txt_no.Text();
                fm.FlowJson = JsonConvert.SerializeObject(vs);
            }
            else
            {
                fm = new FlowModel
                {
                    CreateTime = GlobalUnitils.GetNowTime(DateTime.Now),
                    FlowName = txt_option.Text(),
                    FlowNO = txt_no.Text(),
                    Status = 0,
                    FlowJson = JsonConvert.SerializeObject(vs)
                };
            }
            flag_Save = true;
            Close();
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
        private List<Options> vs = new List<Options>();
        private void Btn_Add_Click(object sender, RoutedEventArgs e)
        {
            Point p = new Point(Left, Top);
            AddFlowDataInfo info = new AddFlowDataInfo(p)
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

        private void DataToUI()
        {
            gd_content.Children.Clear();
            gd_content.RowDefinitions.Clear();
            gd_content.ColumnDefinitions.Clear();

            for (int i = 0; i < vs.Count; i++)
            {
                DataToGridView(vs[i], i + 1);
            }
        }

        private void DataToGridView(Options df, int index)
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
                    GridViewUnitls.AddCell(gd_content, i, index.ToString());
                }
                else if (i == 1)
                {
                    GridViewUnitls.AddCell(gd_content, i, df.OptionName);
                }
                else if (i == 2)
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
            Options df = dev.obj as Options;
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
            Options df = dev.obj as Options;
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
            AddFlowDataInfo info = new AddFlowDataInfo(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            info.isUpdate = true;
            info.ofm = vs[index];
            _ = info.ShowDialog();
            if (info.flag_Save)
            {
                vs[index] = info.ofm;
                DataToUI();
            }
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            txt_option.SetUI("输入流程名称");
            txt_no.SetUI("输入流程编号");

            Btn_Add.SetUI("#E8F4FF", "#97D4FF", "\\Image\\add.png", "#1990FF", "增加");
            Btn_Save.SetUI("#E8F4FF", "#97D4FF", "\\Image\\save.png", "#1990FF", "保存");
            Btn_Close.SetUI("#E8F4FF", "#97D4FF", "\\Image\\语音关闭.png", "#1990FF", "关闭");

            if (isUpdate)
            {
                txt_option.SetText(fm.FlowName);
                vs = JsonConvert.DeserializeObject<List<Options>>(fm.FlowJson);
                DataToUI();
            }
        }
    }
}
