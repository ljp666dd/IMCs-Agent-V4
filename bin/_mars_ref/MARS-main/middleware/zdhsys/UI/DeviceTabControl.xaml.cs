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
using System.Windows.Navigation;
using System.Windows.Shapes;
using zdhsys.Bean;
using zdhsys.Control;
using zdhsys.Popup;
using zdhsys.Unitils;

namespace zdhsys.UI
{
    /// <summary>
    /// DeviceTabControl.xaml 的交互逻辑
    /// </summary>
    public partial class DeviceTabControl : UserControl
    {
        public DeviceTabControl()
        {
            InitializeComponent();

            btn_add.Click += Btn_add_Click;
            btn_del.Click += Btn_del_Click;
            btn_search.Click += Btn_search_Click;
            btn_clear.Click += Btn_clear_Click;
            pg.SetNum(0);
            pg.Click += Pg_Click;
            sb.Click += Sb_Click;
        }

        private void Sb_Click(object sender, RoutedEventArgs e)
        {
            if (!sb.flagLeft)
            {
                Point p = GetParentWindows.GetPoint(this);
                RobotInfo robot = new RobotInfo(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                _ = robot.ShowDialog();
            }
        }

        private void Pg_Click(object sender, RoutedEventArgs e)
        {
            Btn_search_Click(null, null);
        }

        private void Btn_clear_Click(object sender, RoutedEventArgs e)
        {
            txt_option.SetText("");
            dpStart.Text = "";
            dpEnd.Text = "";
        }

        private List<DeviceInfoModel> vs = new List<DeviceInfoModel>();
        private List<DeviceInfoModel> vs2 = new List<DeviceInfoModel>();

        private void Btn_search_Click(object sender, RoutedEventArgs e)
        {
            int index = cbb_status.SelectedIndex;
            string start = "";
            if (!string.IsNullOrEmpty(dpStart.Text))
            {
                start = GlobalUnitils.GetNowTime(DateTime.Parse(dpStart.Text + " 00:00:00")).ToString();
            }
            string end = "";
            if (!string.IsNullOrEmpty(dpEnd.Text))
            {
                end = GlobalUnitils.GetNowTime(DateTime.Parse(dpEnd.Text + " 23:59:59")).ToString();
            }
            vs = SqlHelper.GetDeviceInfoBy(txt_option.Text(), index, start, end);
            //刷新数量
            pg.SetNum(vs.Count);
            int limit = (pg.page - 1) * pg.getPageNum();
            int page = pg.getPageNum();
            vs2.Clear();
            for (int i = limit; i < limit + page; i++)
            {
                if (i + 1 > vs.Count) { break; }
                vs2.Add(vs[i]);
            }
            //把JSON转为对象
            for (int i = 0; i < vs2.Count; i++)
            {
                vs2[i].Dfms = JsonConvert.DeserializeObject<List<DeviceFieldsModel>>(vs2[i].FieldJson);
            }
            DataToUI();
        }

        /// <summary>
        /// 批量删除
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_del_Click(object sender, RoutedEventArgs e)
        {
            List<CheckBox> cbs = GridViewUnitls.FindCheckBoxesInGrid(gd_content);
            int num = 0;
            for (int i = 0; i < cbs.Count; i++)
            {
                if ((bool)cbs[i].IsChecked)
                {
                    num++;
                }
            }
            if (num > 0)
            {
                Point p = GetParentWindows.GetPoint(this);
                DeleteInfo info = new DeleteInfo(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                info.SetUI(":" + num + "个设备");
                _ = info.ShowDialog();
                if (info.flagDelete)
                {
                    for (int i = 0; i < cbs.Count; i++)
                    {
                        if ((bool)cbs[i].IsChecked)
                        {
                            DeviceInfoModel dgi = cbs[i].Tag as DeviceInfoModel;
                            SqlHelper.Delete(dgi);
                        }
                    }
                    Btn_search_Click(null, null);
                }
            }
        }

        private void Btn_add_Click(object sender, RoutedEventArgs e)
        {
            Point p = GetParentWindows.GetPoint(this);
            //if (SqlHelper.CountDeviceGroupInfo() == 0)
            //{
            //    MessageInfo msg = new MessageInfo(p)
            //    {
            //        Width = ActualWidth,
            //        Height = ActualHeight
            //    };
            //    msg.SetUI("请先添加设备组!!!");
            //    _ = msg.ShowDialog();
            //    return;
            //}
            AddDeviceInfo info = new AddDeviceInfo(p.X, p.Y)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            _ = info.ShowDialog();
            //点击保存了。
            if (info.flag_save)
            {
                SqlHelper.Add(info.Dim);
                Btn_search_Click(null, null);
            }
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            txt_option.SetUI("设备名称");

            cbb_status.Items.Add("故障");
            cbb_status.Items.Add("正常");
            cbb_status.Items.Add("全部");
            cbb_status.SelectedIndex = 2;

            btn_search.SetUI("#027AFF", "#027AFF", "\\Image\\搜索.png", "#FFFFFF", "搜索");
            btn_clear.SetUI("#FFFFFF", "#027AFF", "\\Image\\重置.png", "#027AFF", "重置");

            btn_add.SetUI("#E8F4FF", "#97D4FF", "\\Image\\add.png", "#1990FF", "创建");
            btn_del.SetUI("#F2F7FF", "#FF0000", "\\Image\\delete.png", "#FF0000", "批量删除");

            Btn_search_Click(null, null);
        }

        private void DataToUI()
        {
            gd_content.Children.Clear();
            gd_content.RowDefinitions.Clear();
            gd_content.ColumnDefinitions.Clear();
            for (int i = 0; i < vs2.Count; i++)
            {
                DataToGridView(vs2[i]);
            }
        }

        private void DataToGridView(DeviceInfoModel df)
        {
            // 创建新的行 - 固定行高:40
            GridViewUnitls.AddRow(gd_content);

            // 创建并添加新的列到新行
            for (int i = 0; i < gd_header.ColumnDefinitions.Count; i++)
            {
                //当前行，添加1列 -- 列宽对应表头
                GridViewUnitls.AddColumn(gd_content, gd_header, i);

                // 添加对应的控件到新的格子中
                switch (i)
                {
                    case 0:
                        GridViewUnitls.AddCellCheck(gd_content, i, df);
                        break;
                    case 1:
                        GridViewUnitls.AddCell(gd_content, i, df.DeviceName);
                        break;
                    case 2:
                        GridViewUnitls.AddCell(gd_content, i, df.DeviceType == 1 ? "转移类设备" : "反应类设备");
                        break;
                    case 3:
                        GridViewUnitls.AddCell(gd_content, i, df.DeviceId);
                        break;
                    case 4:
                        string deviceGroupName = "";
                        if (!string.IsNullOrEmpty(df.DeviceGroupId))
                        {
                            int em = int.Parse(df.DeviceGroupId);
                            GlobalEnum.UnitDeviceType udt = (GlobalEnum.UnitDeviceType)em;
                            deviceGroupName = udt.ToString();
                        }
                        GridViewUnitls.AddCell(gd_content, i, deviceGroupName);
                        break;
                    case 5:
                        GridViewUnitls.AddCell(gd_content, i, df.TagName);
                        break;
                    case 6:
                        //物体位置
                        string pos = "X:" + df.X + " Y:" + df.Y + " Z:" + df.Z;
                        GridViewUnitls.AddCell(gd_content, i, pos);
                        break;
                    case 7:
                        GridViewUnitls.AddCell(gd_content, i, df.Bottole == 0 ? "大瓶子" : "小瓶子");
                        break;
                    case 8:
                        GridViewUnitls.AddCell(gd_content, i, string.IsNullOrEmpty(df.Remain) ? "-" : df.Remain);
                        break;
                    case 9:
                        //状态
                        GridViewUnitls.AddCellStatus(gd_content, i, df.DeviceStatus == 1 ? "正常" : "故障", df.DeviceStatus == 1 ? GridViewUnitls.ColorHexNormal : GridViewUnitls.ColorHexBreak);
                        break;
                    case 10:
                        GridViewUnitls.AddCell(gd_content, i, string.IsNullOrEmpty(df.PCR) ? "-" : df.Remain);
                        break;
                    case 11:
                        GridViewUnitls.AddCell(gd_content, i, string.IsNullOrEmpty(df.Concentration) ? "-" : df.Concentration);
                        break;
                    case 12:
                        GridViewUnitls.AddCell(gd_content, i, string.IsNullOrEmpty(df.Density) ? "-" : df.Density);
                        break;
                    case 13:
                        GridViewUnitls.AddCell(gd_content, i, string.IsNullOrEmpty(df.Weight) ? "-" : df.Weight);
                        break;
                    case 14:
                        //可操作量
                        string str = "";
                        if (df.Dfms != null)
                        {
                            for (int k = 0; k < df.Dfms.Count; k++)
                            {
                                str += df.Dfms[k].FieldsName + ":" + df.Dfms[k].FieldsContent + " ";
                            }
                        }
                        GridViewUnitls.AddCell(gd_content, i, string.IsNullOrEmpty(str) ? "-" : str);
                        break;
                    case 15:
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
                        _ = gd_content.Children.Add(dev);
                        break;
                    case 16:
                        DevOperation3 dev3 = new DevOperation3
                        {
                            HorizontalAlignment = HorizontalAlignment.Center,
                            VerticalAlignment = VerticalAlignment.Center,
                            obj = df//绑定对象。用于详情，修改，删除之类的。
                        };
                        dev3.Click_Up += Dev3_Click_Up;
                        dev3.Click_Down += Dev3_Click_Down;
                        Grid.SetRow(dev3, gd_content.RowDefinitions.Count - 1);
                        Grid.SetColumn(dev3, i);
                        _ = gd_content.Children.Add(dev3);
                        break;
                    default:
                        break;
                }
            }
            //每行再加一个分割线
            GridViewUnitls.AddSpLine(gd_content, gd_header);
        }

        private void Dev3_Click_Down(object sender, RoutedEventArgs e)
        {
            Console.WriteLine("向下移动");
            DevOperation3 dev = sender as DevOperation3;
            DeviceInfoModel df = dev.obj as DeviceInfoModel;
            List<DeviceInfoModel> vs = SqlHelper.GetDeviceInfo();
            int index = df.Sort + 1;
            DeviceInfoModel df2 = GetDeviceInfoModelBySort(vs, index);
            while(df2 == null)
            {
                index++;
                df2 = GetDeviceInfoModelBySort(vs, index);
                if (df2 != null)
                {
                    break;
                }
                //大于最大排序。说明不能再下去了。
                if (index > vs[vs.Count - 1].Sort)
                {
                    break;
                }
            }
            if (df2 != null)
            {
                // 交换他们两的位置
                df.Sort = df2.Sort;
                df2.Sort = df.Sort;
                SqlHelper.Update(df);
                SqlHelper.Update(df2);
                Btn_search_Click(null, null);
            }
        }

        private DeviceInfoModel GetDeviceInfoModelBySort(List<DeviceInfoModel> vs, int sort)
        {
            for(int i = 0; i < vs.Count; i++)
            {
                if (vs[i].Sort == sort)
                {
                    return vs[i];
                }
            }
            return null;
        }

        private void Dev3_Click_Up(object sender, RoutedEventArgs e)
        {
            Console.WriteLine("向上移动");
            DevOperation3 dev = sender as DevOperation3;
            DeviceInfoModel df = dev.obj as DeviceInfoModel;
            List<DeviceInfoModel> vs = SqlHelper.GetDeviceInfo();
            if (df.Sort > 1)
            {
                int index = df.Sort - 1;
                DeviceInfoModel df2 = GetDeviceInfoModelBySort(vs, index);
                while (df2 == null)
                {
                    if (index > 1)
                    {
                        index--;
                        df2 = GetDeviceInfoModelBySort(vs, index);
                        if (df2 != null)
                        {
                            break;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
                if (df2 == null)
                {
                    // 本身就是最小了。所以不能再向上了。
                    Console.WriteLine("本身就是最小了。所以不能再向上了。");
                }
                else
                {
                    // 交换他们两的位置
                    int sort = df.Sort;
                    df.Sort = df2.Sort;
                    df2.Sort = sort;
                    SqlHelper.Update(df);
                    SqlHelper.Update(df2);
                    Btn_search_Click(null, null);
                }
            }
        }

        private void Dev_Click_Delete(object sender, RoutedEventArgs e)
        {
            DevOperation2 dev = sender as DevOperation2;
            DeviceInfoModel df = dev.obj as DeviceInfoModel;
            Point p = GetParentWindows.GetPoint(this);
            DeleteInfo info = new DeleteInfo(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            info.SetUI("设备:" + df.DeviceName);
            _ = info.ShowDialog();
            if (info.flagDelete)
            {
                SqlHelper.Delete(df);
                Btn_search_Click(null, null);
            }
        }

        private void Dev_Click_Update(object sender, RoutedEventArgs e)
        {
            DevOperation2 dev = sender as DevOperation2;
            DeviceInfoModel df = dev.obj as DeviceInfoModel;
            Point p = GetParentWindows.GetPoint(this);
            AddDeviceInfo info = new AddDeviceInfo(p.X, p.Y)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            info.Dim = df;
            info.isUpdate = true;
            _ = info.ShowDialog();
            //点击保存了。
            if (info.flag_save)
            {
                info.Dim.FieldJson = JsonConvert.SerializeObject(info.Dim.Dfms);
                //修改缓存列表和数据库
                SqlHelper.Update(info.Dim);
                Btn_search_Click(null, null);
            }
        }

        /// <summary>
        /// 全选/取消全选
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void cb_All_Checked(object sender, RoutedEventArgs e)
        {
            bool flag = (bool)cb_All.IsChecked;
            GridViewUnitls.CheckBoxesInGrid(gd_content, flag);
        }

        /// <summary>
        /// 自适应大小
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void UserControl_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            Btn_search_Click(null, null);
        }
    }
}
