using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JsonTool.Helper
{
    public static class ObjectUtils
    {
        public static bool IsNotEmpty(this object obj)
        {
            if (null == obj) return false;

            string str = obj.ToString();
            if (string.IsNullOrEmpty(str)) return false;

            return true;
        }

        public static bool ToBoolean(this object obj, bool defaultVal = false)
        {
            if (obj == null) return defaultVal;

            bool result = defaultVal;
            Boolean.TryParse(obj.ToString(), out result);
            return result;
        }

    }
}
