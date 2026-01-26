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
    public partial class OptionDetails : Window
    {
        public OptionDetails(Point p)
        {
            InitializeComponent();
            Left = p.X;
            Top = p.Y;
            imgClose.Click += ImgClose_Click;
        }

        private void ImgClose_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private List<OptionFieldsModel2> vs = new List<OptionFieldsModel2>();
        public Options opt = null;

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            imgClose.SetUI("\\Image\\语音关闭.png");
            lb_Name.Content = opt.OptionName;
            vs = JsonConvert.DeserializeObject<List<OptionFieldsModel2>>(opt.OptionJson);
            DataToUI();
        }

        private void DataToUI()
        {
            gd_content.Children.Clear();
            gd_content.RowDefinitions.Clear();
            gd_content.ColumnDefinitions.Clear();

            for (int i = 0; i < vs.Count; i++)
            {
                if (vs[i].IsSub)
                {
                    List<SubOptionModel> som = SqlHelper.GetSubOptionModelById(vs[i].SubId);
                    if (som != null && som.Count > 0)
                    {
                        //List<OptionFieldsModel> ofm = JsonConvert.DeserializeObject<List<OptionFieldsModel>>(som[0].Tags);
                        //for (int k = 0; k < ofm.Count; k++)
                        //{
                        //    OptionFieldsModel2 df = new OptionFieldsModel2
                        //    {
                        //        DeviceId = ofm[k].DeviceId,
                        //        Id = ofm[k].Id,
                        //        TagName = ofm[k].TagName,
                        //        TagValue = ofm[k].TagValue,
                        //        IsSub = true,
                        //        SubId = ofm[k].Id
                        //    };
                        //    DataToGridView(df, som[0].SubOptionName);
                        //}
                    }
                }
                else
                {
                    DataToGridView(vs[i], "");
                }
            }
        }

        private void DataToGridView(OptionFieldsModel2 df, string subName)
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
                    GridViewUnitls.AddCell(gd_content, i, df.IsSub ? "是" : "否", df.IsSub ? GridViewUnitls.ColorHexNormal : GridViewUnitls.ColorHexBreak);
                }
                else if (i == 4)
                {
                    GridViewUnitls.AddCell(gd_content, i, subName);
                }
            }
            //每行再加一个分割线
            GridViewUnitls.AddSpLine(gd_content, gd_header);
        }

        private void Window_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            Close();
        }
    }
}
