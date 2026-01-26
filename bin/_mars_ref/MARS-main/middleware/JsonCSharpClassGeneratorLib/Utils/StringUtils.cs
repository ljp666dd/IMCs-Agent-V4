using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Xamasoft.JsonClassGenerator.Utils
{
    internal static class StringUtils
    {

        public static string RemoveText(this string source, String removeText)
        {
            string result = string.Empty;
            int i = source.IndexOf(removeText);
            if (i >= 0)
            {
                result = source.Remove(i, removeText.Length);
                return result;
            }
            return source;
        }

        public static string GetSafeFilename(string arbitraryString)
        {
            var invalidChars = System.IO.Path.GetInvalidFileNameChars();
            var replaceIndex = arbitraryString.IndexOfAny(invalidChars, 0);
            if (replaceIndex == -1) return arbitraryString;

            var r = new StringBuilder();
            var i = 0;

            do
            {
                r.Append(arbitraryString, i, replaceIndex - i);

                switch (arbitraryString[replaceIndex])
                {
                    case '"':
                        r.Append("''");
                        break;
                    case '<':
                        r.Append('\u02c2'); // '˂' (modifier letter left arrowhead)
                        break;
                    case '>':
                        r.Append('\u02c3'); // '˃' (modifier letter right arrowhead)
                        break;
                    case '|':
                        r.Append('\u2223'); // '∣' (divides)
                        break;
                    case ':':
                        r.Append('-');
                        break;
                    case '*':
                        r.Append('\u2217'); // '∗' (asterisk operator)
                        break;
                    case '\\':
                    case '/':
                        r.Append('\u2044'); // '⁄' (fraction slash)
                        break;
                    case '\0':
                    case '\f':
                    case '?':
                        break;
                    case '\t':
                    case '\n':
                    case '\r':
                    case '\v':
                        r.Append(' ');
                        break;
                    default:
                        r.Append('_');
                        break;
                }

                i = replaceIndex + 1;
                replaceIndex = arbitraryString.IndexOfAny(invalidChars, i);
            } while (replaceIndex != -1);

            r.Append(arbitraryString, i, arbitraryString.Length - i);

            return r.ToString();
        }

        public static int StringToInt(string str)
        {
            str = "" + str;
            int val = 0;
            int.TryParse(str, out val);
            return val;
        }

        public static int TryParseInt(this string str)
        {
            str = "" + str;
            int val = 0;
            int.TryParse(str, out val);
            return val;
        }

        public static double TryParseDouble(this string str)
        {
            str = "" + str;
            double val = 0;
            double.TryParse(str, out val);
            return val;
        }

        public static string[] split(this String source, String spString)
        {
            string[] rt = null;
            rt = source.Split(new string[] { spString }, StringSplitOptions.RemoveEmptyEntries);
            return rt;
        }



        /// <summary>
        /// 数据库命名法转化为驼峰命名法
        /// </summary>
        /// <param name="name"></param>
        /// <param name="isBigCamelCaes">is Big Camel Caes</param>
        /// <returns></returns>
        public static string DBNamingToCamelCase(string name, bool isBigCamelCaes = false)
        {
            if (name == null || name.Length == 0) { return ""; }
            if (name.Contains("_"))
            {
                string[] words = name.Split('_');
                string result = string.Empty;
                for (int i = 0; i < words.Length; i++)
                {
                    if (i == 0)
                    {
                        result = words[i];
                    }
                    else
                    {
                        result += UpperCaseFirstLetter(words[i]);
                    }
                }
                if (isBigCamelCaes == true)
                {
                    return UpperCaseFirstLetter(result);
                }
                return result;
            }
            else
            {
                if (isBigCamelCaes == true)
                {
                    return UpperCaseFirstLetter(name);
                }

                return name;
            }
        }
        /// <summary>
        /// 驼峰命名法转化为数据库命名法
        /// </summary>
        /// <param name="name"></param>
        /// <param name="isBigCamelCaes">is Big Camel Caes</param>
        /// <returns></returns>
        public static string CamelCaseToDBnameing(string name, bool isBigCamelCaes = false)
        {
            if (name != null && name.Length > 0)
            {
                if (isBigCamelCaes == true)
                {
                    name = LowerCaseFirstLetter(name);
                }
                char[] array = name.ToCharArray();
                string result = string.Empty;
                for (int i = 0; i < array.Length; i++)
                {
                    if (i == 0)
                    {
                        result += array[i].ToString().ToLower();
                    }
                    else
                    {
                        if (isUpper(array[i]))
                        {
                            result += "_" + array[i].ToString().ToLower();
                        }
                        else if (IsInt(array[i].ToString()))
                        {
                            result += "_" + array[i].ToString();
                        }
                        else
                        {
                            result += array[i].ToString();
                        }
                    }
                }
                return result;
            }
            return "";
        }
        /// <summary>
        /// 是否为整型
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static bool IsInt(string value)
        {
            return Regex.IsMatch(value, @"^[+-]?/d*$");
        }
        /// <summary>
        /// Json 命名改为数据库命名
        /// </summary>
        /// <param name="json"></param>
        /// <returns></returns>
        public static string JsonCamelCaseToDBnameing(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                return null;
            }
            string resultString = string.Empty;
            string pattern = "([\"](\\w+?)[\"][:]{1}?)";
            MatchCollection colls = Regex.Matches(json, pattern);
            for (int i = 0; i < colls.Count; i++)
            {
                json = json.Replace(colls[i].ToString(), DBNamingToCamelCase(colls[i].ToString()));
            }
            return json;
        }
        /// <summary>
        /// Json 命名改为驼峰命名
        /// </summary>
        /// <param name="json"></param>
        /// <returns></returns>
        public static string JsonDBnameingToCamelCase(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                return null;
            }
            string resultString = string.Empty;
            string pattern = "([\"](\\w+?)[\"][:]{1}?)";
            MatchCollection colls = Regex.Matches(json, pattern);
            for (int i = 0; i < colls.Count; i++)
            {
                json = json.Replace(colls[i].ToString(), CamelCaseToDBnameing(colls[i].ToString()));
            }
            return json;
        }



        /// <summary>
        /// 数据库表名转化为类名
        /// </summary>
        /// <param name="dbname"> 数据库表名</param>
        ///  <param name="removeStr"> 要替换的字符</param>
        /// <returns>类名</returns>
        public static string dbNameToClassName(string dbname, string removeStr)
        {
            return upperCaseFirstLetter(dbname, removeStr);
        }

        /// <summary>
        /// 数据库表名转化为类名
        /// </summary>
        /// <param name="dbname"> 数据库表名</param>
        /// <returns>类名</returns>
        public static string dbNameToClassName(string dbname)
        {
            return UpperCaseFirstLetter(dbname);
        }

        /// <summary>
        /// 数据库表名转小写
        /// </summary>
        /// <param name="str"></param>
        /// <param name="removeStr"></param>
        /// <returns></returns>
        public static string dbNameToLowCase(string str, string removeStr)
        {
            if (string.IsNullOrEmpty(str)) return "";

            str = str.Replace(removeStr, "");

            return str.ToLower();
        }


        /// <summary>
        /// 将一个单词的第一个字母变为大写
        /// </summary>
        /// <param name="str"></param>
        /// <returns></returns>
        public static string upperCaseFirstLetter(string str, string removeStr)
        {
            if (string.IsNullOrEmpty(str)) return "";

            str = str.Replace(removeStr, "");

            return str.Substring(0, 1).ToUpper() + str.Substring(1);
        }


        /// <summary>
        /// 将一个单词的第一个字母变为大写
        /// </summary>
        /// <param name="str"></param>
        /// <returns></returns>
        public static string UpperCaseFirstLetter(string str)
        {
            return str.Substring(0, 1).ToUpper() + str.Substring(1);
        }

        /// <summary>
        /// 将一个单词的第一个字母变为小写
        /// </summary>
        /// <param name="str"></param>
        /// <returns></returns>
        public static string LowerCaseFirstLetter(string str)
        {
            return str.Substring(0, 1).ToLower() + str.Substring(1);
        }
        /// <summary>
        /// 判断字符是否为大写字母
        /// </summary>
        /// <param name="c"></param>
        /// <returns></returns>
        public static bool isUpper(char c)
        {
            return c > 'A' && c < 'Z';
        }
    }
}
