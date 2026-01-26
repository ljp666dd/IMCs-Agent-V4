using SQLite;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using zdhsys.Bean;

namespace zdhsys.Unitils
{
    public class SqlHelper
    {
        private static SQLiteConnection db;
        private static readonly string dbName = "zdhsys.db";

        #region 标准数据库操作
        public static void InitSQLite()
        {
            if (File.Exists(dbName))
            {
                db = new SQLiteConnection(dbName);
                return;
            }
            // 如果数据库文件不存在，就创建一个
            db = new SQLiteConnection(dbName);
            //创建表
            _ = db.CreateTable<Options>();
            //设备表
            _ = db.CreateTable<DeviceInfoModel>();
            //设备组表
            _ = db.CreateTable<DeviceGroupInfoModel>();
            //转移类设备
            _ = db.CreateTable<RobotInfoModel>();
            //子配方
            _ = db.CreateTable<SubOptionModel>();
            //流程表
            _ = db.CreateTable<FlowModel>();
        }

        public static void Add(object obj)
        {
            _ = db.Insert(obj);
        }

        public static void Update(object obj)
        {
            _ = db.Update(obj);
        }

        public static void Delete(object obj)
        {
            _ = db.Delete(obj);
        }
        #endregion

        #region 自定义查询

        #region 配方表
        /// <summary>
        /// 配方表
        /// </summary>
        /// <returns></returns>
        public static List<Options> GetOptions()
        {
            return db.Query<Options>("select * from Options");
        }
        /// <summary>
        /// 配方表按条件查询
        /// </summary>
        /// <param name="groupName"></param>
        /// <param name="status"></param>
        /// <param name="startTime"></param>
        /// <param name="endTime"></param>
        /// <returns></returns>
        public static List<Options> GetOptionsInfoBy(string groupName, int status, string startTime, string endTime)
        {
            string sql = "SELECT * FROM Options WHERE 1=1 ";
            if (!string.IsNullOrEmpty(groupName))
            {
                sql += " AND OptionName LIKE '%" + groupName + "%'";
            }
            if (status < 2)
            {
                sql += " AND Status = " + status;
            }
            if (!string.IsNullOrEmpty(startTime))
            {
                sql += " AND CreateTime > " + startTime;
            }
            if (!string.IsNullOrEmpty(endTime))
            {
                sql += " AND CreateTime < " + endTime;
            }
            return db.Query<Options>(sql);
        }

        /// <summary>
        /// 子配方表
        /// </summary>
        /// <returns></returns>
        public static List<SubOptionModel> GetSubOptionModel()
        {
            return db.Query<SubOptionModel>("select * from SubOptionModel");
        }

        /// <summary>
        /// 按ID查询 - 子配方表
        /// </summary>
        /// <returns></returns>
        public static List<SubOptionModel> GetSubOptionModelById(int Id)
        {
            return db.Query<SubOptionModel>("select * from SubOptionModel where Id=" + Id);
        }

        /// <summary>
        /// 子配方表按条件查询
        /// </summary>
        /// <param name="groupName"></param>
        /// <param name="startTime"></param>
        /// <param name="endTime"></param>
        /// <returns></returns>
        public static List<SubOptionModel> GetSubOptionInfoBy(string groupName, string startTime, string endTime)
        {
            string sql = "SELECT * FROM SubOptionModel WHERE 1=1 ";
            if (!string.IsNullOrEmpty(groupName))
            {
                sql += " AND SubOptionName LIKE '%" + groupName + "%'";
            }
            if (!string.IsNullOrEmpty(startTime))
            {
                sql += " AND CreateTime > " + startTime;
            }
            if (!string.IsNullOrEmpty(endTime))
            {
                sql += " AND CreateTime < " + endTime;
            }
            return db.Query<SubOptionModel>(sql);
        }



        #endregion

        #region 流程管理查询

        /// <summary>
        /// 获取所有流程
        /// </summary>
        /// <returns></returns>
        public static List<FlowModel> GetFlowModelInfo()
        {
            return db.Query<FlowModel>("select * from FlowModel");
        }

        /// <summary>
        /// 流程查询按条件
        /// </summary>
        /// <param name="groupName"></param>
        /// <param name="status"></param>
        /// <param name="startTime"></param>
        /// <param name="endTime"></param>
        /// <returns></returns>
        public static List<FlowModel> GetFlowDataInfoBy(string groupName, int status, string startTime, string endTime)
        {
            string sql = "SELECT * FROM FlowModel WHERE 1=1 ";
            if (!string.IsNullOrEmpty(groupName))
            {
                sql += " AND FlowName LIKE '%" + groupName + "%'";
            }
            if (status < 2)
            {
                sql += " AND Status = " + status;
            }
            if (!string.IsNullOrEmpty(startTime))
            {
                sql += " AND CreateTime > " + startTime;
            }
            if (!string.IsNullOrEmpty(endTime))
            {
                sql += " AND CreateTime < " + endTime;
            }
            return db.Query<FlowModel>(sql);
        }

        #endregion

        #region 设备查询
        /// <summary>
        /// 查询所有设备
        /// </summary>
        /// <returns></returns>
        public static List<DeviceInfoModel> GetDeviceInfo()
        {
            List<DeviceInfoModel> vs = db.Query<DeviceInfoModel>("select * from DeviceInfoModel");
            vs = vs.OrderBy(c => c.Sort).ToList();//这里不按ID排序。按Sort排序返回
            return vs;
        }


        /// <summary>
        /// 按条件查询设备
        /// </summary>
        /// <param name="groupName"></param>
        /// <param name="status"></param>
        /// <param name="startTime"></param>
        /// <param name="endTime"></param>
        /// <returns></returns>
        public static List<DeviceInfoModel> GetDeviceInfoBy(string groupName, int status, string startTime, string endTime)
        {
            string sql = "SELECT * FROM DeviceInfoModel WHERE 1=1 ";
            if (!string.IsNullOrEmpty(groupName))
            {
                sql += " AND DeviceName LIKE '%" + groupName + "%'";
            }
            if (status < 2)
            {
                sql += " AND DeviceStatus = " + status;
            }
            if (!string.IsNullOrEmpty(startTime))
            {
                sql += " AND CreateTime > " + startTime;
            }
            if (!string.IsNullOrEmpty(endTime))
            {
                sql += " AND CreateTime < " + endTime;
            }
            List<DeviceInfoModel> vs = db.Query<DeviceInfoModel>(sql);
            vs = vs.OrderBy(c => c.Sort).ToList();//这里不按ID排序。按Sort排序返回
            return vs;
        }

        #endregion

        #region 设备组查询

        /// <summary>
        /// 查询设备组所有
        /// </summary>
        /// <returns></returns>
        public static List<DeviceGroupInfoModel> GetDeviceGroupInfo()
        {
            return db.Query<DeviceGroupInfoModel>("select * from DeviceGroupInfoModel");
        }
        /// <summary>
        /// 按ID查询
        /// </summary>
        /// <param name="Id"></param>
        /// <returns></returns>
        public static List<DeviceGroupInfoModel> GetDeviceGroupInfoById(string Id)
        {
            return db.Query<DeviceGroupInfoModel>("select * from DeviceGroupInfoModel where Id=" + Id);
        }

        /// <summary>
        /// 获取设备组表记录数
        /// </summary>
        /// <returns></returns>
        public static int CountDeviceGroupInfo()
        {
            return db.Query<DeviceGroupInfoModel>("select * from DeviceGroupInfoModel").Count;
        }
        /// <summary>
        /// 按条件查询设备组记录
        /// </summary>
        /// <param name="groupName"></param>
        /// <param name="status"></param>
        /// <param name="startTime"></param>
        /// <param name="endTime"></param>
        /// <returns></returns>
        public static List<DeviceGroupInfoModel> GetDeviceGroupInfoBy(string groupName,int status,string startTime,string endTime)
        {
            string sql = "SELECT * FROM DeviceGroupInfoModel WHERE 1=1 ";
            if (!string.IsNullOrEmpty(groupName))
            {
                sql += " AND DeviceName LIKE '%" + groupName + "%'";
            }
            if (status < 2)
            {
                sql += " AND DeviceStatus = " + status;
            }
            if (!string.IsNullOrEmpty(startTime))
            {
                sql += " AND CreateTime > " + startTime ;
            }
            if (!string.IsNullOrEmpty(endTime))
            {
                sql += " AND CreateTime < " + endTime;
            }
            //sql += " LIMIT " + limitStart + "," + limitEnd;
            return db.Query<DeviceGroupInfoModel>(sql);
        }

        #endregion

        #region 机械臂查询
        /// <summary>
        /// 查询机械臂信息
        /// </summary>
        /// <returns></returns>
        public static List<RobotInfoModel> GetRobotInfo()
        {
            if (db != null)
            {
                return db.Query<RobotInfoModel>("select * from RobotInfoModel");
            }
            return null;
        }

        #endregion

        #endregion

    }
}
