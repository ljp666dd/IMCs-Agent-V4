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
    /// DeviceGroupTabControl.xaml 的交互逻辑
    /// </summary>
    public partial class DeviceGroupTabControl : UserControl
    {
        public DeviceGroupTabControl()
        {
            InitializeComponent();
            InitUI();
        }

        private void InitUI()
        {
            btn_add.Click += Btn_add_Click;
            btn_del.Click += Btn_del_Click;

            btn_search.Click += Btn_search_Click;
            btn_clear.Click += Btn_clear_Click;


            pg.SetNum(0);
            pg.Click += Pg_Click;
        }
        /// <summary>
        /// 清空查询条件
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_clear_Click(object sender, RoutedEventArgs e)
        {
            txt_option.SetText("");
            dpStart.Text = "";
            dpEnd.Text = "";
        }

        /// <summary>
        /// 分页控件被触发
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Pg_Click(object sender, RoutedEventArgs e)
        {
            Btn_search_Click(null, null);
            //DataToPage();
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
                info.SetUI(":" + num + "个设备组");
                _ = info.ShowDialog();
                if (info.flagDelete)
                {
                    for (int i = 0; i < cbs.Count; i++)
                    {
                        if ((bool)cbs[i].IsChecked)
                        {
                            DeviceGroupInfoModel dgi = cbs[i].Tag as DeviceGroupInfoModel;
                            SqlHelper.Delete(dgi);
                        }
                    }
                    Btn_search_Click(null, null);
                }
            }
        }

        private List<DeviceGroupInfoModel> vs = new List<DeviceGroupInfoModel>();
        private List<DeviceGroupInfoModel> vs2 = new List<DeviceGroupInfoModel>();
        /// <summary>
        /// 创建
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_add_Click(object sender, RoutedEventArgs e)
        {
            Point p = GetParentWindows.GetPoint(this);
            AddDeviceGroupInfo info = new AddDeviceGroupInfo(p.X, p.Y)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            _ = info.ShowDialog();
            //点击保存了。
            if (info.flag_save)
            {
                //转换字段列表为JSON
                string json = JsonConvert.SerializeObject(info.Dim.Dfms);
                info.Dim.FieldJson = json;
                //新增到数据库
                SqlHelper.Add(info.Dim);
                Btn_search_Click(null, null);
            }
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            txt_option.SetUI("设备组名称");

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
        /// <summary>
        /// 搜索
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
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
            vs = SqlHelper.GetDeviceGroupInfoBy(txt_option.Text(), index, start, end);
            //刷新数量
            pg.SetNum(vs.Count);

            DataToPage();
        }

        private void DataToPage()
        {
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
        /// <summary>
        /// 显示到GridView
        /// </summary>
        /// <param name="df"></param>
        private void DataToGridView(DeviceGroupInfoModel df)
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
                        GridViewUnitls.AddCell(gd_content, i, df.DeviceType + "");
                        break;
                    case 3:
                        GridViewUnitls.AddCell(gd_content, i, df.DeviceId);
                        break;
                    case 4:
                        GridViewUnitls.AddCell(gd_content, i, df.TagName);
                        break;
                    case 5:
                        //物体位置
                        string pos = "X:" + df.X + " Y:" + df.Y + " Z:" + df.Z;
                        GridViewUnitls.AddCell(gd_content, i, pos);
                        break;
                    case 6:
                        GridViewUnitls.AddCell(gd_content, i, df.Bottole == 0 ? "大瓶子" : "小瓶子");
                        break;
                    case 7:
                        GridViewUnitls.AddCell(gd_content, i, string.IsNullOrEmpty(df.Remain) ? "-" : df.Remain);
                        break;
                    case 8:
                        //状态
                        GridViewUnitls.AddCellStatus(gd_content, i, df.DeviceStatus == 1 ? "正常" : "故障", df.DeviceStatus == 1 ? GridViewUnitls.ColorHexNormal : GridViewUnitls.ColorHexBreak);
                        break;
                    case 9:
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
                    case 10:
                        GridViewUnitls.AddCell(gd_content, i, GlobalUnitils.GetByLong(df.CreateTime).ToString("yyyy-MM-dd HH:mm:ss"));
                        break;
                    case 11:
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
                        break;
                    default:
                        break;
                }
            }
            //每行再加一个分割线
            GridViewUnitls.AddSpLine(gd_content, gd_header);
        }
        /// <summary>
        /// 删除记录
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Dev_Click_Delete(object sender, RoutedEventArgs e)
        {
            DevOperation2 dev = sender as DevOperation2;
            DeviceGroupInfoModel df = dev.obj as DeviceGroupInfoModel;
            Point p = GetParentWindows.GetPoint(this);
            DeleteInfo info = new DeleteInfo(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            info.SetUI("设备组:" + df.DeviceName);
            _ = info.ShowDialog();
            if (info.flagDelete)
            {
                SqlHelper.Delete(df);
                Btn_search_Click(null, null);
            }
        }
        /// <summary>
        /// 修改记录
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Dev_Click_Update(object sender, RoutedEventArgs e)
        {
            DevOperation2 dev = sender as DevOperation2;
            DeviceGroupInfoModel df = dev.obj as DeviceGroupInfoModel;
            Point p = GetParentWindows.GetPoint(this);
            AddDeviceGroupInfo info = new AddDeviceGroupInfo(p.X, p.Y)
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
    }
}
