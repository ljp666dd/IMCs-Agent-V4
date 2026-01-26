using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace zdhsys.Unitils
{
    public class GridViewUnitls
    {
        /// <summary>
        /// 给表格添加一行分割线
        /// </summary>
        /// <param name="gridContent"></param>
        /// <param name="gridHeader"></param>
        public static void AddSpLine(Grid gridContent, Grid gridHeader)
        {
            RowDefinition newRow2 = new RowDefinition
            {
                Height = new GridLength(1)
            };
            // 添加新行到 Grid
            gridContent.RowDefinitions.Add(newRow2);

            for (int i = 0; i < gridHeader.ColumnDefinitions.Count; i++)
            {
                ColumnDefinition newColumn = new ColumnDefinition
                {
                    Width = new GridLength(GetColumnWidth2(gridHeader, i))
                };
                gridContent.ColumnDefinitions.Add(newColumn);
                // 添加对应的控件到新的格子中
                Border border = new Border();
                Color color2 = (Color)ColorConverter.ConvertFromString("#E8E8E8");
                SolidColorBrush brush2 = new SolidColorBrush(color2);
                border.Background = brush2;
                Grid.SetRow(border, gridContent.RowDefinitions.Count - 1);
                Grid.SetColumn(border, i);
                _ = gridContent.Children.Add(border);
            }
        }

        public static void AddSpLine2(Grid gridHeader,int spanColumn, double width)
        {
            RowDefinition newRow2 = new RowDefinition
            {
                Height = new GridLength(1)
            };
            // 添加新行到 Grid
            gridHeader.RowDefinitions.Add(newRow2);
            ColumnDefinition newColumn = new ColumnDefinition
            {
                Width = new GridLength(width)
            };
            gridHeader.ColumnDefinitions.Add(newColumn);
            // 添加对应的控件到新的格子中
            Border border = new Border();
            Color color2 = (Color)ColorConverter.ConvertFromString("#FFFFFF");
            SolidColorBrush brush2 = new SolidColorBrush(color2);
            border.Background = brush2;
            Grid.SetRow(border, gridHeader.RowDefinitions.Count - 1);
            Grid.SetColumn(border, 0);
            Grid.SetColumnSpan(border, spanColumn);
            _ = gridHeader.Children.Add(border);
        }

        /// <summary>
        /// 获取实际列宽
        /// </summary>
        /// <param name="grid"></param>
        /// <param name="columnIndex"></param>
        /// <returns></returns>
        public static double GetColumnWidth2(Grid grid, int columnIndex)
        {
            if (columnIndex >= 0 && columnIndex < grid.ColumnDefinitions.Count)
            {
                ColumnDefinition columnDefinition = grid.ColumnDefinitions[columnIndex];
                return columnDefinition.ActualWidth; // 获取列宽度的值
            }
            else
            {
                return 0; // 或者返回其他默认值或错误处理
            }
        }


        public static void AddCell(Grid gridContent, int index,string content)
        {
            TextBlock newTextBlock = new TextBlock
            {
                Text = content,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center
            };
            Color color = (Color)ColorConverter.ConvertFromString("#333333");
            newTextBlock.Foreground = new SolidColorBrush(color);

            Grid.SetRow(newTextBlock, gridContent.RowDefinitions.Count - 1);
            Grid.SetColumn(newTextBlock, index);
            gridContent.Children.Add(newTextBlock);
        }

        public static void AddCell2(Grid gridContent, int index, string content)
        {
            TextBlock newTextBlock = new TextBlock
            {
                Text = content,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center
            };
            Color color = (Color)ColorConverter.ConvertFromString("#333333");
            newTextBlock.Foreground = new SolidColorBrush(color);

            Border border = new Border
            {
                BorderBrush = Brushes.White,
                BorderThickness = new Thickness(2, 0, 0, 0),
                Child = newTextBlock  // newTextBlock 为之前创建的 TextBlock
            };

            Grid.SetRow(border, gridContent.RowDefinitions.Count - 1);
            Grid.SetColumn(border, index);
            gridContent.Children.Add(border);
        }

        public static void AddCellTextBox(Grid gridContent, int index, string content, object obj,bool flag)
        {
            // 创建 TextBox 控件
            TextBox textBox = new TextBox
            {
                BorderThickness = new Thickness(0),
                Text = content,
                TextAlignment = TextAlignment.Center,
                VerticalContentAlignment = VerticalAlignment.Center,
                Tag = obj,
                IsEnabled = flag
            };
            Color color = (Color)ColorConverter.ConvertFromString("#333333");
            textBox.Foreground = new SolidColorBrush(color);
            Grid.SetRow(textBox, gridContent.RowDefinitions.Count - 1);
            Grid.SetColumn(textBox, index);
            gridContent.Children.Add(textBox);
        }

        public static void AddCell(Grid gridContent, int index, string content, int spanColumn)
        {
            TextBlock newTextBlock = new TextBlock
            {
                Text = content,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center
            };
            Color color = (Color)ColorConverter.ConvertFromString("#333333");
            newTextBlock.Foreground = new SolidColorBrush(color);

            Border border = new Border
            {
                BorderBrush = Brushes.White,
                BorderThickness = new Thickness(2, 0, 0, 0),
                Child = newTextBlock  // newTextBlock 为之前创建的 TextBlock
            };

            Grid.SetRow(border, gridContent.RowDefinitions.Count - 1);
            Grid.SetColumn(border, index);
            Grid.SetColumnSpan(border, spanColumn);
            gridContent.Children.Add(border);
        }

        public static void AddCell(Grid gridContent, int index, string content,string hex)
        {
            TextBlock newTextBlock = new TextBlock
            {
                Text = content,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center
            };
            Color color = (Color)ColorConverter.ConvertFromString(hex);
            newTextBlock.Foreground = new SolidColorBrush(color);
            Grid.SetRow(newTextBlock, gridContent.RowDefinitions.Count - 1);
            Grid.SetColumn(newTextBlock, index);
            gridContent.Children.Add(newTextBlock);
        }

        public static string ColorHexNormal = "#00BC32";
        public static string ColorHexBreak = "#FF0909";


        public static void AddCellStatus(Grid gridContent, int index, string content, string hex)
        {
            //拼装一下，在前面加一个点。
            content = "·" + new string('\u0020', 3) + content;
            TextBlock newTextBlock = new TextBlock
            {
                Text = content,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center
            };
            Color color = (Color)ColorConverter.ConvertFromString(hex);
            newTextBlock.Foreground = new SolidColorBrush(color);
            Grid.SetRow(newTextBlock, gridContent.RowDefinitions.Count - 1);
            Grid.SetColumn(newTextBlock, index);
            gridContent.Children.Add(newTextBlock);
        }

        /// <summary>
        /// 按表头，添加1列
        /// </summary>
        /// <param name="gridContent"></param>
        /// <param name="gridHeader"></param>
        /// <param name="index"></param>
        public static void AddColumn(Grid gridContent, Grid gridHeader,int index)
        {
            ColumnDefinition newColumn = new ColumnDefinition
            {
                Width = new GridLength(GridViewUnitls.GetColumnWidth2(gridHeader, index))
            };
            gridContent.ColumnDefinitions.Add(newColumn);
        }
        /// <summary>
        /// 添加指定宽度的列
        /// </summary>
        /// <param name="gridContent"></param>
        /// <param name="width"></param>
        public static void AddColumn2(Grid gridContent, double width)
        {
            ColumnDefinition newColumn = new ColumnDefinition
            {
                Width = new GridLength(width)
            };
            gridContent.ColumnDefinitions.Add(newColumn);
        }

        public static void AddRow(Grid gridContent)
        {
            RowDefinition newRow = new RowDefinition
            {
                Height = new GridLength(40)
            };
            // 添加新行到 Grid
            gridContent.RowDefinitions.Add(newRow);
        }
        public static void AddRow(Grid gridContent, double width)
        {
            RowDefinition newRow = new RowDefinition
            {
                Height = new GridLength(width)
            };
            // 添加新行到 Grid
            gridContent.RowDefinitions.Add(newRow);
        }

        public static void AddCellCheck(Grid gridContent, int index,object obj)
        {
            CheckBox check = new CheckBox
            {
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                Tag = obj
            };
            Grid.SetRow(check, gridContent.RowDefinitions.Count - 1);
            Grid.SetColumn(check, index);
            _ = gridContent.Children.Add(check);
        }

        public static List<CheckBox> FindCheckBoxesInGrid(Grid grid)
        {
            List<CheckBox> checkboxes = new List<CheckBox>();

            foreach (UIElement element in grid.Children)
            {
                if (element is CheckBox checkbox)
                {
                    checkboxes.Add(checkbox);
                }
                //else if (element is Grid childGrid)
                //{
                //    checkboxes.AddRange(FindCheckBoxesInGrid(childGrid));
                //}
            }
            return checkboxes;
        }

        public static void CheckBoxesInGrid(Grid grid, bool flag)
        {
            for (int i = 0; i < grid.Children.Count; i++)
            {
                UIElement element = grid.Children[i];
                if (element is CheckBox checkbox)
                {
                    checkbox.IsChecked = flag;
                    grid.Children[i] = checkbox;
                }
            }
        }

    }
}
