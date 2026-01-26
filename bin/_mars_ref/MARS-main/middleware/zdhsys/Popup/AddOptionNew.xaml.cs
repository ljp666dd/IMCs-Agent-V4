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
    public partial class AddOptionNew : Window
    {
        public AddOptionNew(Point p)
        {
            InitializeComponent();

            Left = p.X;
            Top = p.Y;

            Btn_Close.Click += Btn_Close_Click;
            Btn_Save.Click += Btn_Save_Click;
            Btn_See.Click += Btn_See_Click;
        }
        /// <summary>
        /// 把流程+配方=生成指令JSON
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_See_Click(object sender, RoutedEventArgs e)
        {
            SeeJsons sJson = new SeeJsons();
            List<List<List<PrintCmd>>> pc = sJson.getJson(fm.Id, Log());
            Point p = new Point(Left, Top);
            SeeJson see = new SeeJson(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            see.pc = pc;
            _ = see.ShowDialog();
        }

        public FlowModel fm = null;
        private List<string> vs = new List<string>();
        public bool isUpdate = false;
        public bool isSee = false;
        public bool flag_Save = false;
        public Options opt = null;
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
                opt = new Options
                {
                    CreateTime = GlobalUnitils.GetNowTime(DateTime.Now)
                };
            }
            opt.OptionJson = json;
            opt.OptionName = txt_option.Text();
            opt.Status = cb.SelectedIndex;
            opt.FlowId = fm.Id;
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
                        //Console.WriteLine("index=" + dfs.index + " " + dfs.dim.DeviceName + " " + dfs.dfm.FieldsName + ":" + text);
                        //这里要合并同一个设备的各各字段
                        int index = vs.FindIndex(x => x.Index == dfs.index);
                        //如果已经存在，就修改对应的字段属性就好了。
                        if (index >= 0)
                        {
                            //int index2 = vs[index].Dfms.FindIndex(x => x.FieldsName == dfs.dfm.FieldsName);
                            //DeviceFieldsModel dfm = JsonConvert.DeserializeObject<DeviceFieldsModel>(JsonConvert.SerializeObject(vs[index].Dfms[index2]));
                            //dfm.FieldsContent = text;
                            //vs[index].Dfms[index2] = dfm;
                            //DevInfo dd = vs[index];
                            int index2 = vs[index].Dfms.FindIndex(x => x.FieldsName == dfs.dfm.FieldsName);
                            //Console.WriteLine($"vs.index={index}:{index2} {dfs.dfm.FieldsName}:{text}");
                            vs[index].Dfms[index2].FieldsContent = text;
                            //Console.WriteLine("修改: "+ dd.DeviceName + " name="+dd.Dfms[index2].FieldsName + ":"+dd.Dfms[index2].FieldsContent);
                        }
                        else
                        {
                            //DeviceInfoModel dim = dfs.dim;
                            DevInfo dim = new DevInfo
                            {
                                Index = dfs.index,
                                Id = dfs.dim.Id,
                                // 这里不能直接=dfs.dim.Dfms 因为会出现引用问题。到时候修改dfms的值，整个vs里面所有的dfms值都会改变。
                                // 这里做一个深度拷贝的处理。复制并创建一个新的Dfms内存块
                                Dfms = JsonConvert.DeserializeObject<List<DeviceFieldsModel>>(JsonConvert.SerializeObject(dfs.dim.Dfms))
                                // Dfms = dfs.dim.Dfms.ToList()
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
            //Console.WriteLine(json2);
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
            Btn_See.SetUI("#E8F4FF", "#97D4FF", "\\Image\\see.png", "#1990FF", "查看JSON");

            _ = cb.Items.Add("正常");
            _ = cb.Items.Add("作废");
            cb.SelectedIndex = 0;

            if (isUpdate)
            {
                cb.SelectedIndex = opt.Status;
                txt_option.SetText(opt.OptionName);
                List<FlowModel> fl = SqlHelper.GetFlowModelInfo();
                fm = fl.Find(x => x.Id == opt.FlowId);
                vs = JsonConvert.DeserializeObject<List<string>>(opt.OptionJson);
                DataToUI();
                DataToGridViewHeader2();
            }
            else
            {
                DataToGridViewHeader();
            }
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

        List<FlowInfo> infos = null;

        private DeviceInfoModel getBaoGui()
        {
            DeviceInfoModel dm = new DeviceInfoModel
            {
                Id = 0,
                DeviceName = "包硅反应",
                Dfms = new List<DeviceFieldsModel>()
            };
            DeviceFieldsModel dmf = new DeviceFieldsModel
            {
                FieldsName = "",
                FieldsContent = ""
            };
            dm.Dfms.Add(dmf);
            return dm;
        }

        private void DataToGridViewHeader()
        {
            infos = JsonConvert.DeserializeObject<List<FlowInfo>>(fm.FlowJson);
            List<DeviceInfoModel> vs = SqlHelper.GetDeviceInfo();
            for (int i = 0; i < vs.Count; i++)
            {
                vs[i].Dfms = JsonConvert.DeserializeObject<List<DeviceFieldsModel>>(vs[i].FieldJson);
            }
            List<DeviceInfoModel> vs2 = new List<DeviceInfoModel>();
            for (int i = 0; i < infos.Count; i++)
            {
                if (infos[i].tempTxt == "包硅反应")
                {
                    DeviceInfoModel dm = getBaoGui();
                    vs2.Add(dm);
                    continue;
                }
                for (int k = 0; k < vs.Count; k++)
                {
                    if (i < infos.Count - 1 && infos[i].DeviceId == vs[k].Id)
                    {
                        vs2.Add(vs[k]);
                        break;
                    }
                    if (i == infos.Count - 1)
                    {
                        if (infos[i].DeviceId == vs[k].Id)
                        {
                            vs2.Add(vs[k]);
                            break;
                        }
                        
                    }
                }
                if (i == infos.Count - 1)
                {
                    for (int k = 0; k < vs.Count; k++)
                    {
                        if (infos[i].DeviceId2 == vs[k].Id)
                        {
                            vs2.Add(vs[k]);
                            break;
                        }
                    }
                }
            }
            vs = vs2;
            int count = 0;
            for (int i = 0; i < vs.Count; i++)
            {
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
                //添加对应的控件到新的格子中
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

            for (int k = 0; k < 12; k++)
            {
                GridViewUnitls.AddRow(gd_content);
                GridViewUnitls.AddColumn2(gd_content, width);
                GridViewUnitls.AddCell2(gd_content, 0, k + 1 + "号瓶");
                // 创建并添加新的列到新行
                for (int i = 1; i < count; i++)
                {
                    GridViewUnitls.AddColumn2(gd_content, width);
                    DeviceFields dfm = GetByIndex2(vs, i);
                    string FieldsContent = dfm.dfm.FieldsContent;
                    // 如果是修改的话，就把里面的值取出来。对应显示
                    if (isUpdate)
                    {
                        List<DevInfo> devs = devList[k];
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
        }

        private void DataToGridViewHeader2()
        {
            List<DeviceInfoModel> vs = SqlHelper.GetDeviceInfo();
            for (int i = 0; i < vs.Count; i++)
            {
                vs[i].Dfms = JsonConvert.DeserializeObject<List<DeviceFieldsModel>>(vs[i].FieldJson);
            }
            List<DeviceInfoModel> vs2 = new List<DeviceInfoModel>();
            for (int i = 0; i < devList[0].Count; i++)
            {
                DevInfo dif = devList[0][i];
                if (dif.Id == 0)
                {
                    DeviceInfoModel dm = getBaoGui();
                    vs2.Add(dm);
                }
                else
                {
                    DeviceInfoModel dm = vs.Find(x => x.Id == dif.Id);
                    if (dm != null)
                    {
                        vs2.Add(dm);
                    }
                }
            }
            vs = vs2;
            int count = 0;
            for (int i = 0; i < vs.Count; i++)
            {
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
                //添加对应的控件到新的格子中
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

            for (int k = 0; k < 12; k++)
            {
                GridViewUnitls.AddRow(gd_content);
                GridViewUnitls.AddColumn2(gd_content, width);
                GridViewUnitls.AddCell2(gd_content, 0, k + 1 + "号瓶");
                // 创建并添加新的列到新行
                for (int i = 1; i < count; i++)
                {
                    GridViewUnitls.AddColumn2(gd_content, width);
                    DeviceFields dfm = GetByIndex2(vs, i);
                    string FieldsContent = dfm.dfm.FieldsContent;
                    // 如果是修改的话，就把里面的值取出来。对应显示
                    if (isUpdate)
                    {
                        List<DevInfo> devs = devList[k];
                        foreach (DevInfo dev in devs)
                        {
                            //用下标。不能用设备ID。 设备是可以重复的。
                            if (dev.Index == dfm.index)
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
                            dim = vs[i],
                            index = i
                        };
                        return df;
                    }
                }
            }
            return null;
        }

    }
}
