// Google Sheet 接收程式（2026-10-06 版）
// 舊格式（main.py SPY訊號、短擠壓等）：照舊加一行喺第一頁
// 新格式（模擬交易 v2）：寫入「現時持倉」「每日總覽」「買賣紀錄」「每月成績表」分頁

function doPost(e) {
  try {
    var data = JSON.parse(e.postData.contents);
    if (data.v == 2) {
      return handleV2(data);
    }
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
    sheet.appendRow([
      data.date,
      data.ticker,
      data.vix,
      data.strategy,
      data.ev,
      data.pivot,
      data.stop_loss,
      "",
      ""
    ]);
    return ContentService.createTextOutput("Success");
  } catch (error) {
    return ContentService.createTextOutput("Error: " + error.message);
  }
}

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
    if (d.action == "ping") {
      return ContentService.createTextOutput("v2-ok");
    }
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
    return ContentService.createTextOutput("ok");
  } catch (error) {
    return ContentService.createTextOutput("Error: " + error.message);
  } finally {
    lock.releaseLock();
  }
}
