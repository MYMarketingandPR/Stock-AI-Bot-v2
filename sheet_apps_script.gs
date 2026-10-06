// ================================================================
// Google Sheet 新格式（模擬交易分頁）— 2026-10-06
// 做法：
//   1. 喺你現有嘅 doPost(e) 函數入面，最頂（第一行）加呢兩行：
//
//        var _d = JSON.parse(e.postData.contents);
//        if (_d.v == 2) return handleV2(_d);
//
//   2. 將下面全部（由 doGet 開始）貼喺 Apps Script 最底。
//   3. 部署 → 管理部署 → 揀現有嗰個 → 鉛筆（編輯）→ 版本揀「新版本」→ 部署。（網址唔會變）
// 舊格式（main.py SPY訊號等）完全唔受影響。
// ================================================================

function doGet(e) {
  if (e && e.parameter && e.parameter.ping == "v2") {
    return ContentService.createTextOutput("v2-ok");
  }
  return ContentService.createTextOutput("ok");
}

function handleV2(d) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    var sh = ss.getSheetByName(d.tab);
    if (!sh) {
      sh = ss.insertSheet(d.tab);
    }
    if (sh.getLastRow() == 0 && d.header) {
      sh.appendRow(d.header);
      sh.setFrozenRows(1);
      sh.getRange(1, 1, 1, d.header.length).setFontWeight("bold").setBackground("#e8f0fe");
    }
    if (d.action == "replace_group" && sh.getLastRow() > 1) {
      var last = sh.getLastRow();
      var vals = sh.getRange(2, d.key_col, last - 1, 1).getValues();
      for (var i = vals.length - 1; i >= 0; i--) {
        if (d.keys.indexOf(String(vals[i][0])) >= 0) {
          sh.deleteRow(i + 2);
        }
      }
    }
    if (d.rows && d.rows.length > 0) {
      var start = sh.getLastRow() + 1;
      sh.getRange(start, 1, d.rows.length, d.rows[0].length).setValues(d.rows);
    }
  } finally {
    lock.releaseLock();
  }
  return ContentService.createTextOutput("ok");
}
