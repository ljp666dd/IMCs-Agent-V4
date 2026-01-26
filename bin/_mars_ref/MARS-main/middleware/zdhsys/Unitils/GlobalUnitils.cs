using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace zdhsys.Unitils
{
    public class GlobalUnitils
    {
        public static DateTime GetByLong(long timestamp)
        {
            //DateTime dateTime = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddMilliseconds(timestamp);
            DateTime startTime = TimeZone.CurrentTimeZone.ToLocalTime(new DateTime(1970, 1, 1)); // 当地时区
            DateTime dt = startTime.AddSeconds(timestamp);
            return dt;
        }
        public static string GetByLong2(long timestamp)
        {
            //DateTime dateTime = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddMilliseconds(timestamp);
            DateTime startTime = TimeZone.CurrentTimeZone.ToLocalTime(new DateTime(1970, 1, 1)); // 当地时区
            DateTime dt = startTime.AddSeconds(timestamp);
            return dt.ToString("yyyy-MM-dd HH:mm:ss");
        }

        public static long GetNowTime(DateTime dateTime)
        {
            //DateTime dateTime = DateTime.Now; // 示例 DateTime
            //DateTime epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
            //long timestamp = (long)(dateTime.ToUniversalTime() - epoch).TotalSeconds;

            //long timestamp = (dateTime.ToUniversalTime().Ticks - 621355968000000000) / 10000000;
            //Console.WriteLine(timestamp);

            DateTime startTime = TimeZone.CurrentTimeZone.ToLocalTime(new DateTime(1970, 1, 1)); // 当地时区
            long timeStamp = (long)(dateTime - startTime).TotalSeconds; // 相差秒数

            //Console.WriteLine(timeStamp);
            return timeStamp;
        }

        public static Point GetScreenPosition(FrameworkElement element)
        {
            // 获取自定义控件在屏幕上的位置
            Point position = element.PointToScreen(new Point(0, 0));
            return position;
        }
        /// <summary>
        /// 生成随机数
        /// </summary>
        /// <returns></returns>
        public static long GetRandomLongNumber()
        {
            byte[] bytes = new byte[8];
            using (RNGCryptoServiceProvider rng = new RNGCryptoServiceProvider())
            {
                rng.GetBytes(bytes);
            }
            return BitConverter.ToInt64(bytes, 0);
        }
    }
}
