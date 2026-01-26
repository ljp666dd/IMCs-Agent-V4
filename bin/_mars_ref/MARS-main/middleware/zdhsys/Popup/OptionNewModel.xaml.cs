using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using zdhsys.Bean;
using zdhsys.Unitils;

namespace zdhsys.Popup
{
    /// <summary>
    /// AddOption.xaml 的交互逻辑
    /// </summary>
    public partial class OptionNewModel : Window
    {
        public OptionNewModel(Point p)
        {
            InitializeComponent();

            Left = p.X;
            Top = p.Y;

            Btn_Close.Click += Btn_Close_Click;
            Btn_Save.Click += Btn_Save_Click;
            Btn_See.Click += Btn_See_Click;
        }

        public string jsonOption = "";
        public bool isUpdate = false;

        private void Btn_See_Click(object sender, RoutedEventArgs e)
        {
            string ret = Log();
            Point p = new Point(Left, Top);
            if (SqlHelper.CountDeviceGroupInfo() == 0)
            {
                MessageInfo2 msg = new MessageInfo2(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                msg.SetUI(ret);
                _ = msg.ShowDialog();
                return;
            }
        }

        public SubOptionModel opt = null;

        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            UpdateDevice();
            Close();
        }

        private void UpdateDevice()
        {
            List<DeviceFields> df = new List<DeviceFields>();
            // 遍历 Grid 中的每一行
            for (int row = 0; row < gd_content.RowDefinitions.Count; row++)
            {
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
                        Console.WriteLine(dfs.dim.DeviceName + " " + dfs.dfm.FieldsName + ":" + text);

                        int index1 = df.FindIndex(x => x.dim.Id == dfs.dim.Id);
                        if (index1 >= 0)
                        {
                            int index2 = df[index1].dim.Dfms.FindIndex(x => x.Id == dfs.dfm.Id);
                            df[index1].dim.Dfms[index2].FieldsContent = text;
                        }
                        else
                        {
                            int index2 = dfs.dim.Dfms.FindIndex(x => x.Id == dfs.dfm.Id);
                            dfs.dim.Dfms[index2].FieldsContent = text;
                            df.Add(dfs);
                        }
                    }
                }
            }
            for (int i = 0; i < df.Count; i++)
            {
                DeviceInfoModel dim = df[i].dim;
                string json = JsonConvert.SerializeObject(dim);
                //Console.WriteLine(json);
                //Console.WriteLine("+++++++++++++++++++++++++++++++++++++++++++++");
                dim.FieldJson = JsonConvert.SerializeObject(dim.Dfms);
                SqlHelper.Update(dim);
            }
        }

        private string Log()
        {
            string ret = "";
            List<DeviceFields> df = new List<DeviceFields>();
            // 遍历 Grid 中的每一行
            for (int row = 0; row < gd_content.RowDefinitions.Count; row++)
            {
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
                        Console.WriteLine(dfs.dim.DeviceName + " " + dfs.dfm.FieldsName + ":" + text);

                        int index1 = df.FindIndex(x => x.dim.Id == dfs.dim.Id);
                        if (index1 >= 0)
                        {
                            int index2 = df[index1].dim.Dfms.FindIndex(x => x.Id == dfs.dfm.Id);
                            df[index1].dim.Dfms[index2].FieldsContent = text;
                        }
                        else
                        {
                            int index2 = dfs.dim.Dfms.FindIndex(x => x.Id == dfs.dfm.Id);
                            dfs.dim.Dfms[index2].FieldsContent = text;
                            df.Add(dfs);
                        }
                    }
                }
            }
            for (int i = 0; i < df.Count; i++)
            {
                DeviceInfoModel dim = df[i].dim;
                ret += dim.DeviceId + "(";
                for(int k = 0; k < dim.Dfms.Count; k++)
                {
                    if (k == dim.Dfms.Count - 1)
                    {
                        ret += dim.Dfms[k].FieldsContent;
                    }
                    else
                    {
                        ret += dim.Dfms[k].FieldsContent + ",";
                    }
                }
                if (i == df.Count - 1)
                {
                    ret += ");";
                }
                else
                {
                    ret += "),";
                }
            }
            return ret;
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }


        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            Btn_See.SetUI("#E8F4FF", "#97D4FF", "\\Image\\see.png", "#1990FF", "查看字符");
            Btn_Save.SetUI("#E8F4FF", "#97D4FF", "\\Image\\save.png", "#1990FF", "保存");
            Btn_Close.SetUI("#E8F4FF", "#97D4FF", "\\Image\\语音关闭.png", "#1990FF", "关闭");

            DataToGridViewHeader();
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
            GridViewUnitls.AddRow(gd_content);
            GridViewUnitls.AddColumn2(gd_content, width);
            GridViewUnitls.AddCell2(gd_content, 0, "配方模板");
            // 创建并添加新的列到新行
            if (!isUpdate)
            {
                for (int i = 1; i < count; i++)
                {
                    GridViewUnitls.AddColumn2(gd_content, width);
                    //DeviceFieldsModel dfm = GetByIndex(vs, i);
                    DeviceFields dfm = GetByIndex2(vs, i);
                    // 添加对应的控件到新的格子中
                    GridViewUnitls.AddCellTextBox(gd_content, i, dfm.dfm.FieldsContent, dfm, true);
                }
            }
            else
            {
                for (int i = 1; i < count; i++)
                {
                    GridViewUnitls.AddColumn2(gd_content, width);
                    //DeviceFieldsModel dfm = GetByIndex(vs, i);
                    DeviceFields dfm = GetByIndex2(vs, i);
                    // 添加对应的控件到新的格子中
                    string ret = GetOptionValue(dfm);
                    GridViewUnitls.AddCellTextBox(gd_content, i, ret, dfm, true);
                }
            }
        }

        private string GetOptionValue(DeviceFields dfm)
        {
            if (string.IsNullOrEmpty(jsonOption))
            {
                return "";
            }
            jsonOption = jsonOption.Replace(";", "");
            MatchCollection matches = Regex.Matches(jsonOption, @"(\w+)\(([^)]+)\)");

            foreach (Match match in matches)
            {
                string id = match.Groups[1].Value;
                string value = match.Groups[2].Value;
                Console.WriteLine($"id={id} value={value}");
                string[] vs = value.Split(',');
                if (dfm.dim.DeviceId == id)
                {
                    int index = dfm.dim.Dfms.FindIndex(x => x.Id == dfm.dfm.Id);
                    if (vs.Length > index)
                    {
                        return vs[index];
                    }
                }
            }
            return "";
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
