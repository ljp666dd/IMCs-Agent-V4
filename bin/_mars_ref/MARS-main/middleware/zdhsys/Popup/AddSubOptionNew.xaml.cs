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
using zdhsys.entity;
using zdhsys.Unitils;

namespace zdhsys.Popup
{
    /// <summary>
    /// AddOption.xaml 的交互逻辑
    /// </summary>
    public partial class AddSubOptionNew : Window
    {
        public AddSubOptionNew(Point p)
        {
            InitializeComponent();

            Left = p.X;
            Top = p.Y;

            Btn_Close.Click += Btn_Close_Click;
            Btn_Save.Click += Btn_Save_Click;
        }

        private List<string> vs = new List<string>();
        public bool isUpdate = false;
        public bool isSee = false;
        public bool flag_Save = false;
        public SubOptionModel opt = null;
        private List<List<DevInfo>> devList = new List<List<DevInfo>>();

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
            string json = Log();
            if (string.IsNullOrEmpty(json))
            {
                info.SetUI("配方内容不能为空");
                _ = info.ShowDialog();
                return;
            }

            if (!isUpdate)
            {
                opt = new SubOptionModel
                {
                    CreateTime = GlobalUnitils.GetNowTime(DateTime.Now)
                };
            }
            opt.OptionJson = json;
            opt.SubOptionName = txt_option.Text();
            opt.Status = cb.SelectedIndex;
            flag_Save = true;
            Close();
        }

        private string Log()
        {
            List<string> vsStr = new List<string>();
            // 遍历 Grid 中的每一行
            for (int row = 0; row < gd_content.RowDefinitions.Count; row++)
            {
                List<DevInfo> vs = new List<DevInfo>();
                // 遍历 Grid 中的每一列
                for (int column = 0; column < gd_content.ColumnDefinitions.Count; column++)
                {
                    // 获取当前行和列上的子元素
                    UIElement element = gd_content.Children
                        .Cast<UIElement>()
                        .FirstOrDefault(e => Grid.GetRow(e) == row && Grid.GetColumn(e) == column);

                    // 检查子元素是否是 TextBox，并获取其值
                    if (element is TextBox textBox)
                    {
                        string text = textBox.Text;
                        DeviceFields dfs = textBox.Tag as DeviceFields;
                        //Console.WriteLine(dfs.dim.DeviceName + " " + dfs.dfm.FieldsName + ":" + text);
                        //这里要合并同一个设备的各各字段
                        int index = vs.FindIndex(x => x.Id == dfs.dim.Id);
                        //如果已经存在，就修改对应的字段属性就好了。
                        if (index >= 0)
                        {
                            DevInfo dd = vs[index];
                            int index2 = dd.Dfms.FindIndex(x => x.FieldsName == dfs.dfm.FieldsName);
                            dd.Dfms[index2].FieldsContent = text;
                            //Console.WriteLine("修改: "+ dd.DeviceName + " name="+dd.Dfms[index2].FieldsName + ":"+dd.Dfms[index2].FieldsContent);
                        }
                        else
                        {
                            //DeviceInfoModel dim = dfs.dim;
                            DevInfo dim = new DevInfo
                            {
                                Id = dfs.dim.Id,
                                Dfms = dfs.dim.Dfms
                            };
                            int index2 = dim.Dfms.FindIndex(x => x.FieldsName == dfs.dfm.FieldsName);
                            DeviceFieldsModel dfm = dim.Dfms[index2];
                            dfm.FieldsContent = text;
                            dim.Dfms[index2] = dfm;
                            vs.Add(dim);
                            //Console.WriteLine("新增: " + dim.DeviceName + " name=" + dim.Dfms[index2].FieldsName + ":" + dim.Dfms[index2].FieldsContent);
                        }
                    }
                }
                string json = JsonConvert.SerializeObject(vs);
                vsStr.Add(json);
            }
            string json2 = JsonConvert.SerializeObject(vsStr);
            Console.WriteLine(json2);
            return json2;
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }


        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            txt_option.SetUI("输入配方名称");
            Btn_Save.SetUI("#E8F4FF", "#97D4FF", "\\Image\\save.png", "#1990FF", "保存");
            Btn_Close.SetUI("#E8F4FF", "#97D4FF", "\\Image\\语音关闭.png", "#1990FF", "关闭");

            cb.Items.Add("正常");
            cb.Items.Add("作废");
            cb.SelectedIndex = 0;

            if (isUpdate)
            {
                cb.SelectedIndex = opt.Status;
                txt_option.SetText(opt.SubOptionName);
                vs = JsonConvert.DeserializeObject<List<string>>(opt.OptionJson);
                DataToUI();
            }
            DataToGridViewHeader();
            if (isSee)
            {
                Btn_Save.IsEnabled = false;
                txt_option.IsEnabled = false;
            }
        }

        private void DataToUI()
        {
            gd_content.Children.Clear();
            gd_content.RowDefinitions.Clear();
            gd_content.ColumnDefinitions.Clear();

            for (int i = 0; i < vs.Count; i++)
            {
                List<DevInfo> devs = JsonConvert.DeserializeObject<List<DevInfo>>(vs[i]);
                devList.Add(devs);
            }
        }

        private void DataToGridViewHeader()
        {
            List<DeviceInfoModel> vs = SqlHelper.GetDeviceInfo();
            int count = 0;
            for (int i = 0; i < vs.Count; i++)
            {
                vs[i].Dfms = JsonConvert.DeserializeObject<List<DeviceFieldsModel>>(vs[i].FieldJson);
                count += vs[i].Dfms.Count;
            }
            count++;
            double widthMax = 1300;
            double width = widthMax / count;
            int Index = 0;
            // 创建新的行- 固定行高:40
            GridViewUnitls.AddRow(gd_header);
            GridViewUnitls.AddColumn2(gd_header, width);
            GridViewUnitls.AddCell2(gd_header, 0, "设备ID");
            // 创建并添加新的列到新行
            for (int i = 0; i < vs.Count; i++)
            {
                for (int k = 0; k < vs[i].Dfms.Count; k++)
                {
                    Index++;
                    GridViewUnitls.AddColumn2(gd_header, width);
                }
                // 添加对应的控件到新的格子中
                GridViewUnitls.AddCell(gd_header, Index - vs[i].Dfms.Count + 1, vs[i].DeviceId, vs[i].Dfms.Count);
            }
            GridViewUnitls.AddSpLine2(gd_header, count, width);
            // 创建新的行- 固定行高:40
            GridViewUnitls.AddRow(gd_header);
            GridViewUnitls.AddColumn2(gd_header, width);
            GridViewUnitls.AddCell2(gd_header, 0, "设备");
            Index = 0;
            // 创建并添加新的列到新行
            for (int i = 0; i < vs.Count; i++)
            {
                for (int k = 0; k < vs[i].Dfms.Count; k++)
                {
                    Index++;
                    GridViewUnitls.AddColumn2(gd_header, width);
                }
                // 添加对应的控件到新的格子中
                GridViewUnitls.AddCell(gd_header, Index - vs[i].Dfms.Count + 1, vs[i].DeviceName, vs[i].Dfms.Count);
            }
            GridViewUnitls.AddSpLine2(gd_header, count, width);
            // 创建新的行- 固定行高:40
            GridViewUnitls.AddRow(gd_header);
            GridViewUnitls.AddColumn2(gd_header, width);
            GridViewUnitls.AddCell2(gd_header, 0, "标签");
            // 创建并添加新的列到新行
            for (int i = 1; i < count; i++)
            {
                GridViewUnitls.AddColumn2(gd_header, width);
                DeviceFieldsModel dfm = GetByIndex(vs, i);
                // 添加对应的控件到新的格子中
                GridViewUnitls.AddCell2(gd_header, i, dfm.FieldsName);
            }
            GridViewUnitls.AddSpLine2(gd_header, count, width);
            // 创建新的行- 固定行高:40
            GridViewUnitls.AddRow(gd_header);
            GridViewUnitls.AddColumn2(gd_header, width);
            GridViewUnitls.AddCell2(gd_header, 0, "配方模板");
            // 创建并添加新的列到新行
            for (int i = 1; i < count; i++)
            {
                GridViewUnitls.AddColumn2(gd_header, width);
                DeviceFieldsModel dfm = GetByIndex(vs, i);
                // 添加对应的控件到新的格子中
                GridViewUnitls.AddCell2(gd_header, i, dfm.FieldsContent);
            }

            //大瓶子
            GridViewUnitls.AddRow(gd_content);
            GridViewUnitls.AddColumn2(gd_content, width);
            GridViewUnitls.AddCell2(gd_content, 0, "大瓶");
            // 创建并添加新的列到新行
            for (int i = 1; i < count; i++)
            {
                GridViewUnitls.AddColumn2(gd_content, width);
                DeviceFields dfm = GetByIndex2(vs, i);
                string FieldsContent = dfm.dfm.FieldsContent;
                // 如果是修改的话，就把里面的值取出来。对应显示
                if (isUpdate)
                {
                    List<DevInfo> devs = devList[0];
                    foreach (DevInfo dev in devs)
                    {
                        if (dev.Id == dfm.dim.Id)
                        {
                            foreach (DeviceFieldsModel dfml in dev.Dfms)
                            {
                                if (dfml.Id == dfm.dfm.Id)
                                {
                                    FieldsContent = dfml.FieldsContent;
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
                // 添加对应的控件到新的格子中
                GridViewUnitls.AddCellTextBox(gd_content, i, FieldsContent, dfm, !isSee);
            }

        }

        private DeviceFieldsModel GetByIndex(List<DeviceInfoModel> vs, int index)
        {
            int count = 0;
            for(int i = 0; i < vs.Count; i++)
            {
                for(int k = 0; k < vs[i].Dfms.Count; k++)
                {
                    count++;
                    if(count == index)
                    {
                        return vs[i].Dfms[k];
                    }
                }
            }
            return null;
        }

        private DeviceFields GetByIndex2(List<DeviceInfoModel> vs, int index)
        {
            int count = 0;
            for (int i = 0; i < vs.Count; i++)
            {
                for (int k = 0; k < vs[i].Dfms.Count; k++)
                {
                    count++;
                    if (count == index)
                    {
                        DeviceFields df = new DeviceFields
                        {
                            dfm = vs[i].Dfms[k],
                            dim = vs[i]
                        };
                        return df;
                    }
                }
            }
            return null;
        }

    }
}
